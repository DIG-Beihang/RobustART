import json
import yaml
import torch
import numpy as np
from .base_evaluator import Evaluator, Metric



class AdvRobustEvaluator(Evaluator):
    def __init__(self):
        super(AdvRobustEvaluator, self).__init__()

    def parse_line(line):
        res = []
        for i in range(0, len(line)):
            if line[i] == ':':
                for j in range(i+2, len(line)):
                    if line[j] == ',' or line[j] == '}':
                        res.append(line[i+2:j])
                        break
        return res[0], res[1]

    def eval(self, clean_path, adv_path):
        cnt_before_att = 0
        cnt_after_att = 0
        f_att = open(adv_path)
        f_clean = open(clean_path)
        lines_att = f_att.readlines()
        lines_clean = f_clean.readlines()
        for ind in range(0, 50000):
            res1_clean, res2_clean = self.parse_line(lines_clean[ind])
            res1_att, res2_att = self.parse_line(lines_att[ind])
            if res1_clean == res2_clean:
                cnt_before_att = cnt_before_att + 1
                if res1_att == res2_att:
                    cnt_after_att = cnt_after_att + 1
        AR = cnt_after_att / cnt_before_att * 100
        print('Clean Acc: {}, Adversarial Robustness: {}'.format(cnt_before_att / 50000 * 100, AR))
        return AR
