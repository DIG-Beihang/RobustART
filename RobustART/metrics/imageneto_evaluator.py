import json
import yaml
import numpy as np
from .base_evaluator import Evaluator, Metric
import RobustART.metrics.calibration_tools as calibration_tools



class ImageNetOEvaluator(Evaluator):
    def __init__(self):
        super(ImageNetOEvaluator, self).__init__()
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

    def eval(self, res_file_in=None, res_file_out=None):
        assert res_file_in is not None and res_file_out is not None
        confidence_in = []
        correct_in = []
        num_correct_in = 0
        confidence_out = []
        correct_out = []
        num_correct_out = 0

        with open(res_file_in) as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            confidence_in += obj['confidence']
            correct_in += obj['correct']
            num_correct_in += obj['num_correct']

        with open(res_file_out) as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            confidence_out += obj['confidence']
            correct_out += obj['correct']
            num_correct_out += obj['num_correct']

        in_score = -np.array(confidence_in)
        out_score = -np.array(confidence_out)

        aurocs, auprs, fprs = [], [], []
        measures = calibration_tools.get_measures(out_score, in_score)
        aurocs = measures[0]
        auprs = measures[1]
        fprs = measures[2]

        result_dict = {'AUPR': (100 * auprs)}
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
        self.metric.update({'Mean': mean})
        self.metric.set_cmp_key('Mean')
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
