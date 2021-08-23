from .imagenet_evaluator import ImageNetEvaluator
from .custom_evaluator import CustomEvaluator
from .multiclass_evaluator import MultiClsEvaluator
from .imagenetc_evaluator import ImageNetCEvaluator


def build_evaluator(cfg):
    evaluator = {
        'custom': CustomEvaluator,
        'imagenet': ImageNetEvaluator,
        'multiclass': MultiClsEvaluator,
        'imagenetc': ImageNetCEvaluator,

    }[cfg['type']]
    return evaluator(**cfg['kwargs'])
