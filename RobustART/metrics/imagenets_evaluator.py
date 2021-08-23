import json
import yaml
import torch
import numpy as np
from .base_evaluator import Evaluator, Metric



class ImageNetSEvaluator(Evaluator):
    def __init__(self):
        super(ImageNetSEvaluator, self).__init__()
        self.metric = Metric()

    def load_res(self, res_file):
        """
        Load results from file.
        """
        res_dict = {}
        with open(res_file) as f:
            lines = f.readlines()
        for line in lines:
            info = json.loads(line)
            for key in info.keys():
                if key not in res_dict.keys():
                    res_dict[key] = [info[key]]
                else:
                    res_dict[key].append(info[key])
        return res_dict

    def eval(self, res_file, decoder_type='pil', resize_type='pil-bilinear'):
        topk = [1]
        res_dict = self.load_res(res_file)
        pred = torch.from_numpy(np.array(res_dict['score']))
        label = torch.from_numpy(np.array(res_dict['label']))
        num = pred.size(0)
        maxk = max(topk)
        _, pred = pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))
        res = {}
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / num)
            res.update({[decoder_type, resize_type]: acc.item()})

        self.metric.update(res)
        return res

    def get_mean(self):
        result_dict = self.metric.metric
        res = []
        for key, item in result_dict:
            res += [item]
        mean = np.mean(res)
        return {'Mean': mean}

    def get_std(self):
        result_dict = self.metric.metric
        res = []
        for key, item in result_dict:
            res += [item]
        std = np.std(res)
        return {'Std.': std}

    def clear(self):
        self.metric.metric = {}

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(
            name, help='subcommand for ImageNet of Top-1/5 accuracy metric')
        subparser.add_argument('--config', dest='config', required=True,
                               help='settings of classification in yaml format')
        subparser.add_argument('--res_file', required=True, action='append',
                               help='results file of classification')

        return subparser

    @classmethod
    def from_args(cls, args):
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        kwargs = config['data']['evaluator']['kwargs']

        return cls(**kwargs)
