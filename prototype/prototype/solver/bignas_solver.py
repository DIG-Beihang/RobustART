import argparse
import random
import copy
import math
import time
import datetime
import torch
import pprint
import os
import json

from .cls_solver import ClsSolver
from prototype.prototype.model import model_entry
from prototype.prototype.utils.dist import link_dist, DistModule
from prototype.prototype.utils.misc import makedir, accuracy, load_state_model, mixup_data, mix_criterion, cutmix_data
from prototype.prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
try:
    from prototype.prototype.spring import ClsController
except ModuleNotFoundError:
    print('spring.nas not detected yet, install spring module first.')


class BigNASSolver(ClsSolver):

    def __init__(self, config_file):
        super(BigNASSolver, self).__init__(config_file)
        self.logger.info(f'model network structure: {pprint.pformat(self.model)}')
        self.controller = ClsController(self.config.bignas)
        self.controller.set_supernet(self.model)
        self.controller.set_logger(self.logger)
        self.path.bignas_path = os.path.join(self.path.root_path, 'bignas')
        makedir(self.path.bignas_path)
        self.controller.set_path(self.path.bignas_path)
        if self.controller.distiller_weight > 0 and 'inplace' not in self.controller.distiller_type:
            self.build_teacher()
        else:
            self.controller.set_teacher(None)
        self.controller.init_distiller()

    def build_teacher(self):
        teacher_model = model_entry(self.controller.config.distiller.model)
        ckpt = torch.load(self.controller.config.kd.model.loadpath, 'cpu')

        teacher_model.cuda()
        teacher_model = DistModule(teacher_model, self.config.dist.sync)

        load_state_model(teacher_model, ckpt)
        self.controller.set_teacher(teacher_model)
        self.logger.info(f'teacher network structure: {pprint.pformat(teacher_model)}')

    def train(self):

        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()

        if self.controller.valid_before_train:
            self.evaluate_specific_subnets(start_step, total_step)

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
            if self.mixup < 1.0:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
            # cutmix
            if self.cutmix > 0.0:
                input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)

            # clear gradient
            self.optimizer.zero_grad()

            for curr_subnet_num in range(self.controller.sample_subnet_num):
                subnet_seed = int('%d%.3d' % (curr_step, curr_subnet_num))
                random.seed(subnet_seed)
                subnet_settings, sample_mode = self.controller.adjust_model(curr_step, curr_subnet_num)

                input = self.controller.adjust_input(input, curr_subnet_num, sample_mode)
                self.controller.subnet_log(curr_subnet_num, input, subnet_settings)

                # before forward, teacher mode should be adjusted
                self.controller.adjust_teacher(input, curr_subnet_num)
                # forward
                logits = self.model(input)
                # mixup
                if self.mixup < 1.0 or self.cutmix > 0.0:
                    loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                    loss /= self.dist.world_size
                else:
                    loss = self.criterion(logits, target) / self.dist.world_size

                # calculate distiller loss
                mimic_loss = self.controller.get_distiller_loss(sample_mode) / self.dist.world_size
                loss += mimic_loss

                # measure accuracy and record loss
                prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

                reduced_loss = loss.clone()
                reduced_prec1 = prec1.clone() / self.dist.world_size
                reduced_prec5 = prec5.clone() / self.dist.world_size

                self.meters.losses.reduce_update(reduced_loss)
                self.meters.top1.reduce_update(reduced_prec1)
                self.meters.top5.reduce_update(reduced_prec5)

                # compute and update gradient
                    loss.backward()

            # compute and update gradient
                self.model.sync_gradients()
                self.optimizer.step()

            # EMA
            if self.ema is not None:
                self.ema.step(self.model, curr_step=curr_step)
            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.controller.show_subnet_log()

                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                          f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                          f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                          f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                          f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                          f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                          f'LR {current_lr:.6f}\t' \
                          f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            # testing during training
            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                if self.controller.subnet is not None:
                    metrics = self.get_subnet_accuracy(self.controller.subnet.image_size,
                                                       self.controller.subnet.subnet_settings, calib_bn=False)
                else:
                    metrics = self.evaluate_specific_subnets(curr_step, total_step)
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

    def ofa_calib_bn(self, model, image_size):
        self.logger.info('prepare to set running statistics')
        config = copy.deepcopy(self.config.data)
        config.input_size = image_size
        config.test_resize = math.ceil(image_size / 0.875)
        config.train.meta_file = self.controller.calib_meta_file
        config.max_iter = 4096 // self.dist.world_size // config.batch_size
        config.last_iter = 0
        self.calib_data = build_imagenet_train_dataloader(config)
        self.val_data = build_imagenet_test_dataloader(config)
        model = self.controller.reset_subnet_running_statistics(model, self.calib_data['loader'])
        return model

    def get_subnet_accuracy(self, image_size=None, subnet_settings=None, calib_bn=True):
        if image_size is None:
            image_size = self.controller.sample_image_size(sample_mode='random')
        if subnet_settings is None:
            subnet_settings = self.controller.sample_subnet_settings(sample_mode='random')
        else:
            subnet_settings = self.controller.sample_subnet_settings('subnet', subnet_settings)
        if calib_bn:
            self.model = self.ofa_calib_bn(self.model, image_size[3])
        metrics = self.evaluate()
        top1, top5 = round(metrics.metric['top1'], 3), round(metrics.metric['top5'], 3)
        self.logger.info('Subnet with settings: {}\ttop1 {}\ttop5 {}'.format(subnet_settings, top1, top5))
        return metrics

    def get_subnet_latency(self, image_size, subnet_settings, flops):
        onnx_name = self.controller.get_subnet_prototxt(
                image_size=image_size, subnet_settings=subnet_settings,
                flops=flops, onnx_only=False)
        latency = self.controller.get_subnet_latency(onnx_name)
        while not latency:
            time.sleep(1)
            latency = self.controller.get_subnet_latency(onnx_name)
        return latency

    def evaluate_specific_subnets(self, curr_step, total_step):
        valid_log = 'Valid: [%d/%d]' % (curr_step, total_step)
        for setting in self.model.module.subnet_settings:
            for image_size in self.controller.test_image_size_list:
                name = '_'.join(['%s_%s' % (
                    key, '%s' % val) for key, val in setting.items()])
                curr_name = 'R' + str(image_size) + '_' + name
                self.logger.info(curr_name)
                metrics = self.get_subnet_accuracy(image_size, setting)
                top1, top5 = round(metrics.metric['top1'], 3), round(metrics.metric['top5'], 3)
                flops, params, _, _ = self.controller.get_subnet_flops(image_size, setting)
                valid_log += '\t%s (%.3f, %.3f, %.3f, %.3f),' % (curr_name, top1, top5, flops, params)
        self.logger.info(valid_log)
        # return the biggeset model val_loss, prec1, prec5
        return metrics

    # 在image size变化的时候需要重新build到对应的train dataset
    def build_subnet_finetune_dataset(self, image_size, max_iter=None):
        config = copy.deepcopy(self.config.data)
        config.input_size = image_size
        config.test_resize = math.ceil(image_size / 0.875)
        config.last_iter = 0
        if max_iter is None:
            max_iter = config.max_iter
        config.max_iter = max_iter
        self.logger.info('build subnet finetune training dataset with image size {} max_iter {}'.format(
            image_size, max_iter))
        self.train_data = build_imagenet_train_dataloader(config)

    # 测试一个特定的子网，配置从self.subnet里面取
    def evaluate_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None

        self.save_subnet_weight = self.subnet.get('save_subnet_weight', False)
        self.save_subnet_prototxt = self.subnet.get('save_subnet_prototxt', False)
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', False)

        metrics = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        top1, top5 = round(metrics.metric['top1'], 3), round(metrics.metric['top5'], 3)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('Evaluate_subnet\t{}'.format(json.dumps(subnet)))
        self.logger.info('Subnet:\n{}'.format(self.model.module.module_str))
        if self.save_subnet_weight:
            subnet = self.controller.get_subnet_weight(self.subnet.subnet_settings)
            state_dict = {}
            state_dict['model'] = subnet.state_dict()
            ckpt_name = f'{self.path.bignas_path}/ckpt_{flops}.pth.tar'
            torch.save(state_dict, ckpt_name)
        if self.save_subnet_prototxt:
            onnx_name = self.controller.get_subnet_prototxt(self.subnet.image_size, self.subnet.subnet_settings,
                                                            flops, only_onnx=False)
        if self.test_subnet_latency:
            latency = self.controller.get_subnet_latency(onnx_name)
            return latency, params, top1, top5
        return flops, params, top1, top5

    # finetune一个特定的子网，配置从self.subnet里面取
    def finetune_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        self.config.data.last_iter = 0
        metrics = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        top1, top5 = round(metrics.metric['top1'], 3), round(metrics.metric['top5'], 3)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('Before finetune subnet {}'.format(json.dumps(subnet)))
        self.logger.info('Subnet:\n{}'.format(self.model.module.module_str))
        self.build_subnet_finetune_dataset(self.subnet.image_size[3])
        self.train()
        metrics = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings, calib_bn=False)
        top1, top5 = round(metrics.metric['top1'], 3), round(metrics.metric['top5'], 3)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('After finetune subnet {}'.format(json.dumps(subnet)))
        return flops, params, top1, top5

    def sample_multiple_subnet_flops(self):
        self.subnet_dict = self.controller.sample_subnet_lut(test_latency=True)

    def sample_multiple_subnet_accuracy(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        self.subnet_dict = self.controller.sample_subnet_lut(test_latency=False)
        self.sample_with_finetune = self.subnet.get('sample_with_finetune', False)
        self.performance_dict = []
        self.baseline_flops = self.subnet.get('baseline_flops', None)
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', False)

        for k, v in self.subnet_dict.items():
            self.subnet.image_size = v['image_size']
            self.subnet.subnet_settings = v['subnet_settings']
            if self.sample_with_finetune:
                # 重新load超网，这样不会受到前一个子网训练的影响
                loadpath = self.config.model.get('loadpath', None)
                assert loadpath is not None
                state = torch.load(loadpath, map_location='cpu')
                load_state_model(self.model, state['model'])
                # 如果image size变了，需要重新build finetune的dataset
                self.build_subnet_finetune_dataset(self.subnet.image_size[3])
                _, _, v['top1'], v['top5'] = self.finetune_subnet()
                self.logger.info('Sample_subnet_({}) with finetuning\t{}'.format(k, json.dumps(v)))
            else:
                metrics = self.get_subnet_accuracy(v['image_size'], v['subnet_settings'], calib_bn=True)
                v['top1'], v['top5'] = round(metrics.metric['top1'], 3), round(metrics.metric['top5'], 3)
                if 'latency' not in v.keys() and self.test_subnet_latency:
                    latency = self.get_subnet_latency(v['image_size'], v['subnet_settings'], v['flops'])
                    v['latency'] = latency

                self.logger.info('Sample_subnet_({})\t{}'.format(k, json.dumps(v)))
            self.performance_dict.append(v)

        self.get_top10_subnets()
        self.get_pareto_subnets()
        self.get_latency_pareto_subnets()

    def get_top10_subnets(self):
        self.baseline_flops = self.subnet.get('baseline_flops', None)
        if self.baseline_flops is None:
            return
        self.performance_dict = sorted(self.performance_dict, key=lambda x: x['top1'])
        candidate_dict = [_ for _ in self.performance_dict
                          if (_['flops'] - self.baseline_flops) / self.baseline_flops < 0.01]
        if len(candidate_dict) == 0:
            return
        candidate_dict = sorted(candidate_dict, key=lambda x: x['top1'], reverse=True)
        self.logger.info('---------------top10---------------')
        length = 10 if len(candidate_dict) > 10 else len(candidate_dict)
        for c in (candidate_dict[:length]):
            self.logger.info(json.dumps(c))
        self.logger.info('-----------------------------------')

    def get_pareto_subnets(self, key='flops'):
        self.performance_dict = sorted(self.performance_dict, key=lambda x: x[key])
        pareto = []
        for info in self.performance_dict:
            flag = True
            for _ in self.performance_dict:
                if info == _:
                    continue
                if info['top1'] < _['top1'] and info[key] >= _[key]:
                    flag = False
                    break
                if info['top1'] <= _['top1'] and info[key] > _[key]:
                    flag = False
                    break
            if flag:
                pareto.append(info)
        self.logger.info('---------------{} pareto---------------'.format(key))
        for p in pareto:
            self.logger.info(json.dumps(p))
        self.logger.info('---------------------------------------')

    def get_latency_pareto_subnets(self):
        keys = []
        if 'latency' in self.performance_dict[0].keys():
            for k in self.performance_dict[0]['latency']:
                keys.append(k)
        else:
            return
        for key in keys:
            pareto = []
            self.performance_dict = sorted(self.performance_dict, key=lambda x: x['latency'][key])
            for info in self.performance_dict:
                flag = True
                for _ in self.performance_dict:
                    if info == _:
                        continue
                    if info['top1'] < _['top1'] and info['latency'][key] >= _['latency'][key]:
                        flag = False
                        break
                    if info['top1'] <= _['top1'] and info['latency'][key] > _['latency'][key]:
                        flag = False
                        break
                if flag:
                    pareto.append(info)
            self.logger.info('---------------{} pareto---------------'.format(key))
            for p in pareto:
                self.logger.info(json.dumps(p))
            self.logger.info('---------------------------------------')


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Neural archtecture search Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--phase', default='train_supnet')

    args = parser.parse_args()
    # build solver
    solver = BigNASSolver(args.config)
    # evaluate or train
    if args.phase in ['evaluate_subnet', 'finetune_subnet', 'sample_accuracy']:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        if args.phase == 'evaluate_subnet':
            solver.evaluate_subnet()
        elif args.phase == 'finetune_subnet':
            solver.finetune_subnet()
        else:
            solver.sample_multiple_subnet_accuracy()
    elif args.phase == 'train_supnet':
        if solver.config.data.last_iter <= solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')
    elif args.phase == 'sample_flops':
        solver.sample_multiple_subnet_flops()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
