from torch.utils.data import DataLoader

from .datasets import CustomDataset, MultiClassDataset
from .transforms import build_transformer
from .sampler import build_sampler
from .metrics import build_evaluator


def build_custom_dataloader(data_type, cfg_dataset):
    """
    arguments:
        - data_type: 'train', 'test', 'val'
        - cfg_dataset: configurations of dataset
    """
    assert data_type in cfg_dataset
    # build transformer
    image_reader = cfg_dataset[data_type].get('image_reader', {})
    transformer = build_transformer(cfgs=cfg_dataset[data_type]['transforms'],
                                    image_reader=image_reader)
    # build evaluator
    evaluator = None
    if data_type == 'test' and cfg_dataset[data_type].get('evaluator', None):
        evaluator = build_evaluator(cfg_dataset[data_type]['evaluator'])
    # build dataset
    if cfg_dataset['type'] == 'custom':
        CurrDataset = CustomDataset
    elif cfg_dataset['type'] == 'multiclass':
        CurrDataset = MultiClassDataset
    else:
        raise NotImplementedError

    if cfg_dataset['read_from'] == 'osg':
        dataset = CurrDataset(
            root_dir='',
            meta_file=cfg_dataset[data_type]['meta_file'],
            transform=transformer,
            read_from='osg',
            evaluator=evaluator,
            image_reader_type=image_reader.get('type', 'pil'),
            osg_server=cfg_dataset[data_type]['osg_server'],
        )
    else:
        dataset = CurrDataset(
            root_dir=cfg_dataset[data_type]['root_dir'],
            meta_file=cfg_dataset[data_type]['meta_file'],
            transform=transformer,
            read_from=cfg_dataset['read_from'],
            evaluator=evaluator,
            image_reader_type=image_reader.get('type', 'pil')
        )
    # initialize kwargs of sampler
    cfg_dataset[data_type]['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    # build sampler
    sampler = build_sampler(cfg_dataset[data_type]['sampler'], cfg_dataset)
    if data_type == 'train' and cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}
    # build dataloader
    loader = DataLoader(dataset=dataset,
                        batch_size=cfg_dataset['batch_size'],
                        shuffle=False if sampler is not None else True,
                        num_workers=cfg_dataset['num_workers'],
                        pin_memory=cfg_dataset['pin_memory'],
                        sampler=sampler)
    return {'type': data_type, 'loader': loader}
