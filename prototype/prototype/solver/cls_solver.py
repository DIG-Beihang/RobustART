import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
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
    mix_criterion, modify_state, cutmix_data, parse_config
from prototype.prototype.utils.ema import EMA
from prototype.prototype.model import model_entry
from prototype.prototype.optimizer import optim_entry
from prototype.prototype.lr_scheduler import scheduler_entry
from prototype.prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.prototype.data import build_custom_dataloader
from prototype.prototype.loss_functions import LabelSmoothCELoss


class ClsSolver(BaseSolver):

    def __init__(self, config_file):
        self.config_file = config_file
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
            self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
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

        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)
        count_flops(self.model, input_shape=[
                    1, 3, self.config.data.input_size, self.config.data.input_size])


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
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        else:
            self.train_data = build_custom_dataloader('train', self.config.data)

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
            input = input.cuda()
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

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        imagenetc_flag = self.config.data.test.get("imagenet_c", False)
        if imagenetc_flag:

            noise_list = []

            writer = {'noise': {'gaussian_noise': {}, 'shot_noise': {}, 'impulse_noise': {}},
                 'blur': {'defocus_blur': {},
                          'glass_blur': {},
                          'motion_blur': {},
                          'zoom_blur': {}},
                 'weather': {'snow': {}, 'frost': {}, 'fog': {}, 'brightness': {}},
                 'digital': {'contrast': {},
                             'elastic_transform': {},
                             'pixelate': {},
                             'jpeg_compression': {}},
                 'extra': {'speckle_noise': {},
                           'spatter': {},
                           'gaussian_blur': {},
                           'saturate': {}}}
            for noise in writer:
                for noise_type in writer[noise]:
                    for i in range(1, 6):
                        res_file = os.path.join(self.path.result_path,
                                                f'{noise}-{noise_type}-{i}-results.txt.rank{self.dist.rank}')
                        writer[noise][noise_type][i] = open(res_file, 'w')
                        noise_list.append(os.path.join(self.path.result_path,
                                                       f'{noise}-{noise_type}-{i}-results.txt.rank'))
            noise_list = sorted(noise_list)
        else:
            res_file = os.path.join(self.path.result_path, f'results.txt.rank{self.dist.rank}')
            writer = open(res_file, 'w')

        bn_lst = []
        for batch_idx, batch in enumerate(self.val_data['loader']):
            if batch_idx % 10 == 0:
                info_str = f"[{batch_idx}/{len(self.val_data['loader'])}] ";
                info_str += f"{batch_idx * 100 / len(self.val_data['loader']):.6f}%"
                self.logger.info(info_str)


            input = batch['image']
            label = batch['label']
            input = input.cuda()
            label = label.squeeze().view(-1).cuda().long()
            # compute output
            logits= self.model(input)
            # bn_lst.append(bn)
            # if batch_idx % 10 == 0:
            #     torch.save(bn_lst, "./bn_clean.pth")
            #     exit(0)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, batch)
        if imagenetc_flag:
            for noise in writer:
                for noise_type in writer[noise]:
                    for i in range(1, 6):
                        writer[noise][noise_type][i].close()
        else:
            writer.close()
        link.barrier()
        if imagenetc_flag:
            for idx, file_prefix in enumerate(noise_list):
                if idx % self.dist.world_size == self.dist.rank:
                    # print(f"idx: {idx}, rank: {self.dist.rank}, {file_prefix}")
                    self.val_data['loader'].dataset.evaluate(file_prefix)
            link.barrier()
            if self.dist.rank == 0:
                self.val_data['loader'].dataset.merge_eval_res(self.path.result_path)
            metrics = {}
        else:
            if self.dist.rank == 0:
                metrics = self.val_data['loader'].dataset.evaluate(res_file)
                self.logger.info(json.dumps(metrics.metric, indent=2))
            else:
                metrics = {}
        link.barrier()

        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        self.model.train()
        return metrics


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    solver = ClsSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        solver.evaluate()
        if solver.ema is not None:
            solver.ema.load_ema(solver.model)
            solver.evaluate()
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
