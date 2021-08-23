import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import time
import datetime
import torch
import random
import json
import prototype.spring.linklink as link
import torch.nn.functional as F

from .base_solver import BaseSolver
from prototype.prototype.utils.dist import link_dist, DistModule, broadcast_object
from prototype.prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer, mixup_data, \
    mix_criterion, cutmix_data, parse_config
from prototype.prototype.utils.ema import EMA
from prototype.prototype.model import model_entry
from prototype.prototype.optimizer import optim_entry
from prototype.prototype.lr_scheduler import scheduler_entry
from prototype.prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.prototype.data import build_custom_dataloader
from prototype.prototype.loss_functions import LabelSmoothCELoss
from prototype.prototype.model import get_model_robust_baseline, get_model_robust_trick, get_efficient
import traceback
import shutil
import numpy as np
import copy
#from prototype.prototype.utils.user_analysis_helper import send_info

#from prototype.spring.models import SPRING_MODELS_REGISTRY


class IPEvalSolver(BaseSolver):

    def __init__(self, config, model, prefix_name):
        self.prototype_info = EasyDict()
        self.prefix_name = prefix_name
        self.config = config
        self.model = model
        # self.model.cuda()
        self.setup_env()
        self.check_rank()
        # self.build_model()
        # self.build_optimizer()
        # self.build_data()
        # self.build_lr_scheduler()
        #send_info(self.prototype_info)
        count_params(self.model)
        count_flops(self.model, input_shape=[
            1, 3, self.config.data.input_size, self.config.data.input_size])

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.getcwd()
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, self.prefix_name, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, self.prefix_name, 'log.txt'))
        self.logger = get_logger(__name__)
        # self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint

        self.state = {}
        self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def check_rank(self):
        if self.dist.world_size > 7:
            self.logger.warning("If your GPUs are on multi nodes, please use GPU on the same node."
                                " Or a error will occur when load the imagenet-p data")

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)
        count_flops(self.model, input_shape=[
                    1, 3, self.config.data.input_size, self.config.data.input_size])

        # handle fp16

        self.model = DistModule(self.model, self.config.dist.sync)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if 'ema' in self.state:
            self.ema.load_state_dict(self.state['ema'])

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            self.val_data = build_custom_dataloader('test', self.config.data)

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        label_smooth = self.config.get('label_smooth', 0.0)
        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        if label_smooth > 0:
            self.logger.info('using label_smooth: {}'.format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.mixup = self.config.get('mixup', 1.0)
        self.cutmix = self.config.get('cutmix', 0.0)
        self.switch_prob = 0.0
        if self.mixup < 1.0:
            self.logger.info('using mixup with alpha of: {}'.format(self.mixup))
        if self.cutmix > 0.0:
            self.logger.info('using cutmix with alpha of: {}'.format(self.cutmix))
        if self.mixup < 1.0 and self.cutmix > 0.0:
            # the probability of switching mixup to cutmix if both are activated
            self.switch_prob = self.config.get('switch_prob', 0.5)
            self.logger.info('switching between mixup and cutmix with probility of: {}'.format(self.switch_prob))

    def train(self):

        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        for i, batch in enumerate(self.train_data['loader']):
            input = batch['image']
            target = batch['label']
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            target = target.squeeze().cuda().long()
            input = input.cuda().half() if self.fp16 else input.cuda()
            # mixup
            if self.mixup < 1.0 and random.uniform(0, 1) > self.switch_prob:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
            # cutmix
            elif self.cutmix > 0.0:
                input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)
            # forward
            logits = self.model(input)
            # mixup
            if self.mixup < 1.0 or self.cutmix > 0.0:
                loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                loss /= self.dist.world_size
            else:
                loss = self.criterion(logits, target) / self.dist.world_size
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

            reduced_loss = loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)

            # compute and update gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.model.sync_gradients()
            self.optimizer.step()

            # EMA
            if self.ema is not None:
                self.ema.step(self.model, curr_step=curr_step)
            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            # testing during training
            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                metrics = self.evaluate()
                if self.ema is not None:
                    self.ema.load_ema(self.model)
                    ema_metrics = self.evaluate()
                    self.ema.recover(self.model)
                    if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                        metric_key = 'top{}'.format(self.topk)
                        self.tb_logger.add_scalars('acc1_val', {'ema': ema_metrics.metric['top1']}, curr_step)
                        self.tb_logger.add_scalars('acc5_val', {'ema': ema_metrics.metric[metric_key]}, curr_step)

                # testing logger
                if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                    metric_key = 'top{}'.format(self.topk)
                    self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                    self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)

                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    if self.ema is not None:
                        self.state['ema'] = self.ema.state_dict()
                    torch.save(self.state, ckpt_name)

            end = time.time()

    def flip_prob(self, predictions, noise_perturbation=True):
        result = 0
        step_size = 1

        for vid_preds in predictions:
            result_for_vid = []

            for i in range(step_size):
                prev_pred = vid_preds[i]

                for pred in vid_preds[i::step_size][1:]:
                    result_for_vid.append(int(prev_pred != pred))
                    if not noise_perturbation: prev_pred = pred

            result += np.mean(result_for_vid) / len(predictions)

        return result

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        imagenetp_flag = self.config.data.test.get("imagenet_p", False)
        root_dir = self.config.data.test.root_dir

        assert imagenetp_flag, 'This solver is only for imagenet-p'

        perturbation_list = ['brightness', 'motion_blur', 'zoom_blur',
                             'spatter', 'translate', 'rotate', 'tilt', 'scale',
                             'speckle_noise', 'gaussian_blur', 'snow', 'shear', 'shot_noise']

        #perturbation_list = ['gaussian_noise']
        for perturbation in perturbation_list:
            res_file = os.path.join(self.path.result_path, f'{perturbation}-results.txt.rank{self.dist.rank}')
            writer = open(res_file, 'w')


            self.config.data.test.root_dir = os.path.join(root_dir, perturbation)
            self.build_data()

            predictions, ranks = [], []
            for data, target in self.val_data['loader']:
                num_vids = data.size(0)
                data = data.view(-1, 3, 224, 224).cuda()
                output = self.model(data)

                for vid in output.view(num_vids, -1, 1000):
                    predictions = vid.argmax(1).to('cpu').numpy().tolist()
                    res = {'predictions': predictions}
                    writer.write(json.dumps(res, ensure_ascii=False) + '\n')
                    # predictions.append(vid.argmax(1).to('cpu').numpy())
                    #ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])
            #ranks = np.asarray(ranks)

            writer.flush()
            writer.close()
            link.barrier()

            # get fp
            if self.dist.rank == 0:
                prefix = res_file.rstrip('0123456789')
                world_size = link.get_world_size()
                merged_file = prefix.rsplit('.', 1)[0] + '.all'
                merged_fd = open(merged_file, 'w')
                for rank in range(world_size):
                    res_file = prefix + str(rank)
                    assert os.path.exists(res_file), f'No such file or directory: {res_file}'
                    with open(res_file, 'r') as fin:
                        for line_idx, line in enumerate(fin):
                            merged_fd.write(line)
                merged_fd.close()

                pre_res = []
                with open(merged_file) as f:
                    lines = f.readlines()
                for line in lines:
                    one_pre = json.loads(line)['predictions']
                    pre_res.append(np.array(one_pre))

                self.fp = self.flip_prob(pre_res, noise_perturbation=True if 'noise' in perturbation else False)
                self.logger.info(f'Model: {self.prefix_name} Perturbation: {perturbation}')
                self.logger.info('Flipping Prob\t{:.5f}'.format(self.fp))

            link.barrier()




@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save-detail', action='store_true')   # will use over 100G disk pre model
    parser.add_argument('--ckpt-filePath', default='/mnt/lustre/share/robust/ckpt_all')

    args = parser.parse_args()
    # build solver
    config = parse_config(args.config)
    #model_list = config['eval_list']
    model_dict = get_model_robust_baseline()
    model_dict_trick = get_model_robust_trick()
    model_dict.update(model_dict_trick)

    model_dict = get_efficient()
    status = open("status.txt", "a")


    for model_name, model in model_dict.items():
        file_path = args.ckpt_filePath
        ckpt_path = os.path.join(file_path, model_name + '.pth.tar')
        try:
            print('Loading pretrain model for ' + model_name)
            state = torch.load(ckpt_path, 'cpu')
            # state = modify_state(state, EasyDict())
            # for key in list(state['model'].keys()):
            #     if 'module.' in key:
            #         state['model'][key.split('module.')[1]] = state['model'].pop(key)
            model.cuda()
            model = DistModule(model, False)
            load_state_model(model, state['model'])
        except:
            print("Error when load " + model_name)
            print(traceback.format_exc())
            status.write(f"Error when load {model_name}, skip it.\n")
            status.write(traceback.format_exc())
            continue

        if 'efficientnet' in model_name:
            if model_name.split('_')[-1].isdigit():
                input_size = int(model_name.split('_')[-1])
                test_size = int(input_size*256/224)
                config.data.input_size = input_size
                config.data.test_resize = test_size
        else:
            config.data.input_size = 224
            config.data.test_resize = 256

        solver = IPEvalSolver(copy.deepcopy(config), model, model_name)
        # evaluate or train
        solver.evaluate()
        status.write(f"{model_name} done\n")

        # remove detail file to free disk
        if not args.save_detail:
            shutil.rmtree(solver.path.result_path, ignore_errors=True)

    status.close()

if __name__ == '__main__':
    main()
