import prototype.spring.linklink as link

from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam, AdamW  # noqa F401


def optim_entry(config):
    rank = link.get_rank()
    return globals()[config['type']](**config['kwargs'])
