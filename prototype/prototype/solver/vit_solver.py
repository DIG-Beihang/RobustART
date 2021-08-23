import argparse
from .cls_solver import ClsSolver
from .cifar_cls_solver import CifarClsSolver
from prototype.prototype.optimizer import optim_entry
from prototype.prototype.utils.misc import load_state_optimizer
from prototype.prototype.utils.ema import EMA
from prototype.prototype.utils.dist import link_dist


def _build_optimizer(solver):

    opt_config = solver.config.optimizer
    opt_config.kwargs.lr = solver.config.lr_scheduler.kwargs.base_lr
    solver.prototype_info.optimizer = solver.config.optimizer.type

    # make param_groups
    if opt_config.get('no_wd', False):
        fc_nowd = opt_config.no_wd.get('fc', False)
        norm_nowd = opt_config.no_wd.get('norm', False)
        no_wd = solver.model.module.get_param_no_wd(fc=fc_nowd, norm=norm_nowd)

        normal_params = []
        for p in solver.model.parameters():
            wd = True
            for nw in no_wd:
                if p is nw:
                    wd = False
                    break
            if wd:
                normal_params.append(p)
        param_group = [{'params': no_wd, 'weight_decay': 0.0},
                       {'params': normal_params}]
    else:
        param_group = [{'params': solver.model.parameters()}]

    opt_config.kwargs.params = param_group
    solver.optimizer = optim_entry(opt_config)

    if 'optimizer' in solver.state:
        load_state_optimizer(solver.optimizer, solver.state['optimizer'])

    # EMA
    if solver.config.ema.enable:
        solver.config.ema.kwargs.model = solver.model
        solver.ema = EMA(**solver.config.ema.kwargs)
    else:
        solver.ema = None

    if 'ema' in solver.state:
        solver.ema.load_state_dict(solver.state['ema'])


class ViTSolver(ClsSolver):
    def __init__(self, config):
        super(ViTSolver, self).__init__(config)

    def build_optimizer(self):
        _build_optimizer(self)


class CifarViTSolver(CifarClsSolver):
    def __init__(self, config):
        super(CifarViTSolver, self).__init__(config)

    def build_optimizer(self):
        _build_optimizer(self)


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--solver', default='ViTSolver')

    args = parser.parse_args()
    # build solver
    if args.solver == 'ViTSolver':
        solver = ViTSolver(args.config)
    elif args.solver == 'CifarViTSolver':
        solver = CifarViTSolver(args.config)
    else:
        raise ValueError('unhandled solver type: {}'.format(args.solver))

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
