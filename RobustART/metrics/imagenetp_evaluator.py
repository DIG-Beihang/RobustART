import json
import yaml
import torch
import numpy as np
from .base_evaluator import Evaluator, Metric



class ImageNetPEvaluator(Evaluator):
    def __init__(self):
        super(ImageNetPEvaluator, self).__init__()
        self.metric = Metric()

    def load_res(self, res_file):
        """
        Load results from file.
        """
        res_dict = {}
        pre_res = []
        with open(res_file) as f:
            lines = f.readlines()
        for line in lines:
            one_pre = json.loads(line)['predictions']
            pre_res.append(np.array(one_pre))
        return pre_res

    def eval(self, res_file, perturbation=None):
        predictions = self.load_res(res_file)
        noise_perturbation = True if 'noise' in perturbation else False
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
        result_dict = {perturbation: result}
        self.metric.update(result_dict)
        return result_dict

    def get_mean(self):
        result_dict = self.metric.metric
        sum = 0
        idx = 0
        for key, item in result_dict:
            idx += 1
            sum += item
        mean = sum / idx
        return {'Mean': mean}

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
