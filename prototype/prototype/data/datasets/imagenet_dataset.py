import os.path as osp
import json
import requests
import time
import numpy as np

from .base_dataset import BaseDataset
from prototype.prototype.data.image_reader import build_image_reader

import prototype.spring.linklink as link
import random
import torch


class ImageNetDataset(BaseDataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - server_cfg (list): server configurations

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """

    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='mc', evaluator=None, image_reader_type='pil',
                 server_cfg={}):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.initialized = False
        self.use_server = False

        if len(server_cfg) == 0:
            # read from local file
            with open(meta_file) as f:
                lines = f.readlines()

            self.num = len(lines)
            self.metas = []
            for line in lines:
                filename, label = line.rstrip().split()
                self.metas.append({'filename': filename, 'label': label})
        else:
            # read from http server
            self.server_ip = server_cfg['ip']
            self.server_port = server_cfg['port']
            self.use_server = True
            if isinstance(self.server_ip, str):
                self.server_ip = [self.server_ip]
            if isinstance(self.server_port, int):
                self.server_port = [self.server_port]
            assert len(self.server_ip) == len(self.server_port), \
                'length of ips should equal to the length of ports'

            self.num = int(requests.get('http://{}:{}/get_len'.format(
                self.server_ip[0], self.server_port[0])).json())

        super(ImageNetDataset, self).__init__(root_dir=root_dir,
                                              meta_file=meta_file,
                                              read_from=read_from,
                                              transform=transform,
                                              evaluator=evaluator)

    def __len__(self):
        return self.num

    def _load_meta(self, idx):
        if self.use_server:
            while True:
                # random select a server ip
                rdx = np.random.randint(len(self.server_ip))
                r_ip, r_port = self.server_ip[rdx], self.server_port[rdx]
                # require meta information
                try:
                    meta = requests.get('http://{}:{}/get/{}'.format(r_ip, r_port, idx), timeout=500).json()
                    break
                except Exception:
                    time.sleep(0.005)

            return meta
        else:
            return self.metas[idx]

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = osp.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'image_id': idx,
            'label': label,
            'filename': filename
        }
        return item

    def dump(self, writer, output):
        prediction = self.tensor2numpy(output['prediction'])
        label = self.tensor2numpy(output['label'])
        score = self.tensor2numpy(output['score'])

        if 'filename' in output:
            # pytorch type: {'image', 'label', 'filename', 'image_id'}
            filename = output['filename']
            image_id = output['image_id']
            for _idx in range(prediction.shape[0]):
                res = {
                    'filename': filename[_idx],
                    'image_id': int(image_id[_idx]),
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        else:
            # dali type: {'image', 'label'}
            for _idx in range(prediction.shape[0]):
                res = {
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()


class RankedImageNetDataset(BaseDataset):
    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='mc', evaluator=None, image_reader_type='pil',
                 server_cfg={}):
        self.rank = link.get_rank()
        self.world_size = link.get_world_size()
        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.initialized = False
        self.use_server = False

        if len(server_cfg) == 0:
            # read from local file
            ranked_meta = [[] for _ in range(self.world_size)]
            with open(meta_file) as f:
                random.seed(0)
                np.random.seed(0)
                for line in f:
                    filename, label = line.rstrip().split()
                    random_rank = random.randint(0, self.world_size - 1)
                    if self.rank == random_rank:
                        ranked_meta[random_rank].append({'filename': filename, 'label': label})
            self.metas = ranked_meta[self.rank]
            self.num = len(self.metas)

            # balance data length in each subprocess
            ranked_num = [torch.LongTensor([0]) for _ in range(self.world_size)]
            link.allgather(ranked_num, torch.LongTensor([self.num]))
            link.barrier()
            max_num = max([item.item() for item in ranked_num])
            if max_num > self.num:
                diff = max_num - self.num
                self.metas.extend(random.sample(self.metas, diff))
                self.num = len(self.metas)
            else:
                assert self.num == max_num
        else:
            raise NotImplementedError

        super(RankedImageNetDataset, self).__init__(root_dir=root_dir,
                                                    meta_file=meta_file,
                                                    read_from=read_from,
                                                    transform=transform,
                                                    evaluator=evaluator)

    def __len__(self):
        return self.num

    def _load_meta(self, idx):
        if self.use_server:
            raise NotImplementedError
        else:
            return self.metas[idx]

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = osp.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'label': label,
            'image_id': idx,
            'filename': filename
        }
        return item

    def dump(self, writer, output):
        prediction = self.tensor2numpy(output['prediction'])
        label = self.tensor2numpy(output['label'])
        score = self.tensor2numpy(output['score'])

        if 'filename' in output:
            # pytorch type: {'image', 'label', 'filename', 'image_id'}
            filename = output['filename']
            image_id = output['image_id']
            for _idx in range(prediction.shape[0]):
                res = {
                    'filename': filename[_idx],
                    'image_id': int(image_id[_idx]),
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        else:
            # dali type: {'image', 'label'}
            for _idx in range(prediction.shape[0]):
                res = {
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()
