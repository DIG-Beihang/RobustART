import os
import argparse
from easydict import EasyDict
import torch
import spring.linklink as link
import torch.nn.functional as F

from .base_solver import BaseSolver
from prototype.prototype.utils.dist import link_dist, DistModule
from prototype.prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    load_state_model, modify_state, parse_config
from prototype.prototype.model import model_entry
from prototype.prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.prototype.data import build_custom_dataloader
from prototype.prototype.utils.model_config import model_name_dict
from RobustART.noise.add_noise import AddNoise

import numpy as np
import foolbox as fb


def normalize(x, mode='normal', typ=False):
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].cuda()
    var = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].cuda()
    if typ:
        mean = mean.half()
        var = var.half()
    if mode == 'normal':
        return (x - mean) / var
    elif mode == 'inv':
        return x * var + mean



class ClsSolver(BaseSolver):

    def __init__(self, config, prefix):
        self.prototype_info = EasyDict()
        self.config = config
        self.prefix = prefix
        self.fp16 = False
        self.setup_env()
        self.build_model()
        self.build_data()

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.getcwd()
        #self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        #self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, self.prefix)
        #makedir(self.path.save_path)
        #makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        #if self.dist.rank == 0:
        #    self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.result_path, 'log.txt'))
        self.logger = get_logger(__name__)
        # logger already set
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            self.state_src = torch.load(self.config.saver.pretrain.path_src, 'cpu')
            self.state_tgt = torch.load(self.config.saver.pretrain.path_tgt, 'cpu')
            self.logger.info(f"source model: Recovering from {self.config.saver.pretrain.path_src}, keys={list(self.state_src.keys())}")
            self.logger.info(f"target model: Recovering from {self.config.saver.pretrain.path_tgt}, keys={list(self.state_tgt.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model_src = model_entry(self.config.model_src)
        self.model_tgt = model_entry(self.config.model_tgt)
        self.prototype_info.model_src = self.config.model_src.type
        self.prototype_info.model_tgt = self.config.model_tgt.type
        self.model_src.cuda()
        self.model_tgt.cuda()

        count_params(self.model_src)
        count_params(self.model_tgt)
        count_flops(self.model_src, input_shape=[
                    1, 3, self.config.data.input_size, self.config.data.input_size])
        count_flops(self.model_tgt, input_shape=[
                    1, 3, self.config.data.input_size, self.config.data.input_size])

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if self.config.optimizer.get('fp16_normal_bn', False):
                self.logger.info('using normal bn for fp16')
                link.fp16.register_float_module(link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.info('using normal fc for fp16')
                link.fp16.register_float_module(torch.nn.Linear, cast_args=True)
            link.fp16.init()
            self.model.half()

        self.model_src = DistModule(self.model_src, self.config.dist.sync)
        self.model_tgt = DistModule(self.model_tgt, self.config.dist.sync)

        load_state_model(self.model_src, self.state_src['model'])
        load_state_model(self.model_tgt, self.state_tgt['model'])

    def build_data(self):
        self.config.data.last_iter = self.state_tgt['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        else:
            self.train_data = build_custom_dataloader('train', self.config.data)

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            self.val_data = build_custom_dataloader('test', self.config.data)

    #@torch.no_grad()
    def evaluate(self, attack='none', eps=0):  # PGD-Linf, CW-L2, FGSM, AutoAttack
        self.logger.info('Start evaluating on {} attack, eps={}'.format(attack, str(eps)))
        self.model_src.eval()
        self.model_tgt.eval()
        device = "cuda:" + str(int(self.dist.rank%8))
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        f_model = fb.PyTorchModel(self.model_src, bounds=(0, 1), device=device, preprocessing=preprocessing)
        attack_gen = AddNoise(attack)
        attack_gen.set_config(f_model=f_model, eps=eps)

        res_file = os.path.join(self.path.result_path, f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            if self.dist.rank == 0:
                if batch_idx % 10 == 0:
                    info_str = f"[{batch_idx}/{len(self.val_data['loader'])}] ";
                    info_str += f"{batch_idx * 100 / len(self.val_data['loader']):.6f}%"
                    self.logger.info(info_str)
            input = batch['image']
            label = batch['label']
            input = input.cuda().half() if self.fp16 else input.cuda()
            label = label.squeeze().view(-1).cuda().long()

            # if use foolbox, input should not be normalized
            if attack not in ['none']:
                input = normalize(input, 'inv', self.fp16)
                # generate adv examples
                adv_input = attack_gen.add_noise(input, label)
                input = normalize(adv_input.cuda(), typ=self.fp16)

            # compute output
            logits = self.model_tgt(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            self.val_data['loader'].dataset.dump_no_score(writer, batch)

        writer.close()
        link.barrier()
        if self.dist.rank == 0:
            metrics = self.val_data['loader'].dataset.evaluate(res_file)
            #self.logger.info(json.dumps(metrics.metric, indent=2))
        else:
            metrics = {}
        link.barrier()
        # broadcast metrics to other process
        #metrics = broadcast_object(metrics)
        #self.model.train()
        #return metrics


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    # source and target model for adversarial eval
    parser.add_argument('--src_name', required=True, type=str)
    parser.add_argument('--src_path', required=True, type=str)
    parser.add_argument('--tgt_name', required=True, type=str)    
    parser.add_argument('--tgt_path', required=True, type=str) 
    # attack params
    parser.add_argument('--attack', required=True, type=str)
    parser.add_argument('--eps', required=True, type=str)
    
    args = parser.parse_args()
    # modify config
    config = parse_config(args.config)
    config.model_src = model_name_dict[args.src_name.split('--')[0]]
    config.model_tgt = model_name_dict[args.tgt_name.split('--')[0]]
    config.saver.pretrain.path_src = args.src_path
    config.saver.pretrain.path_tgt = args.tgt_path
    config.data.test.evaluator = {}
    if 'efficientnet' in args.src_name.split('--')[0]:
        efficient_imgsize = [224, 240, 260, 300, 380]
        efficient_resize = [256, 274, 298, 342, 434]
        config.data.input_size = efficient_imgsize[int(args.src_name.split('--')[0][-1])]
        config.data.test_resize = efficient_resize[int(args.src_name.split('--')[0][-1])]
    # build solver
    prefix = args.attack + '_' + str(eval(args.eps))[0:min(5, len(str(eval(args.eps))))]
    solver = ClsSolver(config, prefix)
    #solver.evaluate(attack='none', eps=0)
    solver.evaluate(attack=args.attack, eps=eval(args.eps))

if __name__ == '__main__':
    main()
