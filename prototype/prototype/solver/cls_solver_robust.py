import argparse
import time
import datetime
import torch
import torch.nn as nn

from prototype.prototype.utils.dist import link_dist
from prototype.prototype.utils.misc import AverageMeter, accuracy, mixup_data, \
    mix_criterion, cutmix_data
from prototype.prototype.optimizer import FP16RMSprop, FP16SGD, FusedFP16SGD
from prototype.prototype.solver.cls_solver import ClsSolver


class KurtosisWeight:
    def __init__(self, weight_tensor, name, kurtosis_target=1.8, k_mode='avg'):
        self.kurtosis_loss = 0
        self.kurtosis = 0
        self.weight_tensor = weight_tensor
        self.name = name
        self.k_mode = k_mode
        self.kurtosis_target = kurtosis_target

    def fn_regularization(self):
        return self.kurtosis_calc()

    def kurtosis_calc(self):
        mean_output = torch.mean(self.weight_tensor)
        std_output = torch.std(self.weight_tensor)
        kurtosis_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 4))
        self.kurtosis_loss = (kurtosis_val - self.kurtosis_target) ** 2
        self.kurtosis = kurtosis_val

        if self.k_mode == 'avg':
            self.kurtosis_loss = torch.mean((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.mean(kurtosis_val)
        elif self.k_mode == 'max':
            self.kurtosis_loss = torch.max((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.max(kurtosis_val)
        elif self.k_mode == 'sum':
            self.kurtosis_loss = torch.sum((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.sum(kurtosis_val)
        return self.kurtosis


class ClsSolver_Robust(ClsSolver):

    def __init__(self, config_file):
        super(ClsSolver_Robust, self).__init__(config_file)

    def train(self):

        self.pre_train()
        self.kw_weight = self.config.get('kw_weight', 1)
        self.meters.kw_loss = AverageMeter(self.config.saver.print_freq)
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
            if self.mixup < 1.0:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
            # cutmix
            if self.cutmix > 0.0:
                input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)
            # forward
            logits = self.model(input)
            # mixup
            if self.mixup < 1.0 or self.cutmix > 0.0:
                loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                loss /= self.dist.world_size
            else:
                loss = self.criterion(logits, target) / self.dist.world_size

            # remove the first conv weight
            count = 0
            kurtosis_loss = 0
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d) and count != 0:
                    kurtosis = KurtosisWeight(m.weight, name+'.weight')
                    # self.logger.info('kurtosis loss for {}th layer with {}'.format(count, name))
                    kurtosis_loss += kurtosis.fn_regularization() / self.dist.world_size
                if isinstance(m, nn.Conv2d):
                    count += 1
            kw_loss = self.kw_weight * kurtosis_loss / (count-1)
            loss += kw_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

            reduced_loss = loss.clone()
            reduced_kw_loss = kw_loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.kw_loss.reduce_update(reduced_kw_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)

            # compute and update gradient
            self.optimizer.zero_grad()
            if FusedFP16SGD is not None and isinstance(self.optimizer, FusedFP16SGD):
                self.optimizer.backward(loss)
                self.model.sync_gradients()
                self.optimizer.step()
            elif isinstance(self.optimizer, FP16SGD) or isinstance(self.optimizer, FP16RMSprop):

                def closure():
                    self.optimizer.backward(loss, False)
                    self.model.sync_gradients()
                    # check overflow, convert to fp32 grads, downscale
                    self.optimizer.update_master_grads()
                    return loss
                self.optimizer.step(closure)
            else:
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
                self.tb_logger.add_scalar('kw_loss_train', self.meters.kw_loss.avg, curr_step)
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


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    solver = ClsSolver_Robust(args.config)
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
