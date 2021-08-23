import os


model_names = [
    'my_model_A--base'
    'my_model_B--ema'
    ]


def parse_line(line):
    res = []
    for i in range(0, len(line)):
        if line[i] == ':':
            for j in range(i+2, len(line)):
                if line[j] == ',' or line[j] == '}':
                    res.append(line[i+2:j])
                    break
    return res[0] == res[1]


root_path = './'
f_w = open('./trans_result.txt', 'a')
for src_name in model_names:
    f_w.write(src_name + '\n')
    print('parsing {}'.format(src_name))
    for tgt_name in model_names:
        res_path = os.path.join(root_path, src_name+"_To_"+tgt_name, 'fgsm_0.031', 'results.txt.all')
        if os.path.exists(res_path):
            cnt_before = 0
            cnt_after = 0
            res_src = os.path.join('../eval/', src_name, 'none_0', 'results.txt.all')
            res_tgt = os.path.join('../eval/', tgt_name, 'none_0', 'results.txt.all')
            line_trans = open(res_path).readlines()
            line_src = open(res_src).readlines()
            line_tgt = open(res_tgt).readlines()
            if len(line_trans) == 50000 and len(line_src) == 50000 and len(line_tgt) == 50000:
                for ind in range(0, 50000):
                    if parse_line(line_src[ind]) and parse_line(line_tgt[ind]):
                        cnt_before = cnt_before + 1
                        if not parse_line(line_trans[ind]):
                            cnt_after = cnt_after + 1
                f_w.write(str(cnt_after / cnt_before) + '\n')
            else:
                f_w.write('\n')
        else:
            f_w.write('\n')
