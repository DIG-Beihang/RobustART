import json
import yaml
import torch
import numpy as np
from .base_evaluator import Evaluator, Metric



class WorstCaseAdvRobustEvaluator(Evaluator):
    def __init__(self):
        super(WorstCaseAdvRobustEvaluator, self).__init__()

    def parse_line(line):
        res = []
        for i in range(0, len(line)):
            if line[i] == ':':
                for j in range(i+2, len(line)):
                    if line[j] == ',' or line[j] == '}':
                        res.append(line[i+2:j])
                        break
        return res[0], res[1]

    def eval(self, clean_path, multi_adv_result_paths):
        lines_clean = open(clean_path).readlines()
        cnt_before_att = 0
        cnt_after_att = 0
        list_lines_att = []
        for adv_result_path in multi_adv_result_paths:
            lines_att = open(adv_result_path).readlines()
            list_lines_att.append(lines_att)
        for ind in range(0, 50000):
            res1_clean, res2_clean = self.parse_line(lines_clean[ind])
            if res1_clean == res2_clean:
                cnt_before_att = cnt_before_att + 1
                is_correct = 1
                for line_att in list_lines_att:
                    res1_att, res2_att = self.parse_line(line_att[ind])
                    if res1_att != res2_att:
                        is_correct = 0
                if is_correct == 1:
                    cnt_after_att = cnt_after_att + 1
        WCAR = cnt_after_att / cnt_before_att * 100
        print('Worst-Case Adversarial Robustness: {}'.format(WCAR))
        return WCAR
