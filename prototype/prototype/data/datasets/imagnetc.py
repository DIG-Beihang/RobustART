import os.path as osp
import json
import os
from .base_dataset import BaseDataset
from prototype.prototype.data.image_reader import build_image_reader
from prettytable import PrettyTable
import numpy as np
from prototype.prototype.utils.misc import get_logger
from prototype.prototype.data.image_reader import build_image_reader
#from prototype.spring.data import IMAGE_READER, IMAGE_DECODER


class ImageNet_C_Dataset(BaseDataset):
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
        #self.image_reader = IMAGE_READER['mc']()
        #self.image_decoder = IMAGE_DECODER[image_reader_type]()
        self.image_reader = build_image_reader(image_reader_type)
        self.initialized = False
        self.use_server = False
        self.logger = get_logger(__name__)
        # read from local file
        with open(meta_file, "r") as f:
            meta_file = json.load(f)
        assert isinstance(meta_file, dict)
        self.metas = []
        for noise in meta_file:
            for noise_type in meta_file[noise]:
                for severity in meta_file[noise][noise_type]:
                    f_name = meta_file[noise][noise_type][severity]
                    with open(f_name) as f:
                        lines = f.readlines()

                    for line in lines:
                        data = json.loads(line)
                        data.update({
                            "noise": noise,
                            "noise_type": noise_type,
                            "severity": int(severity)
                        })
                        self.metas.append(data)
        self.num = len(self.metas)
        super(ImageNet_C_Dataset, self).__init__(root_dir=root_dir,
                                                 meta_file=meta_file,
                                                 read_from=read_from,
                                                 transform=transform,
                                                 evaluator=evaluator)

    def __len__(self):
        return self.num

    def _load_meta(self, idx):
        return self.metas[idx]

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = osp.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        # Can't use mc
        # img_bytes = self.image_reader(filename)
        # img = self.image_decoder(img_bytes)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'label': label,
            'image_id': idx,
            'filename': filename,
            "noise": curr_meta["noise"],
            "noise_type": curr_meta["noise_type"],
            "severity": curr_meta["severity"]
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
            noise = self.tensor2numpy(output['noise'])
            noise_type = self.tensor2numpy(output['noise_type'])
            severity = self.tensor2numpy(output['severity'])

            for _idx in range(prediction.shape[0]):
                res = {
                    'filename': filename[_idx],
                    'image_id': int(image_id[_idx]),
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }

                writer[noise[_idx]][noise_type[_idx]][int(severity[_idx])].write(
                    json.dumps(res, ensure_ascii=False) + '\n')
                writer[noise[_idx]][noise_type[_idx]
                                    ][int(severity[_idx])].flush()
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

    def evaluate(self, res_file):
        """
        Arguments:
            - res_file (:obj:`str`): filename of result
        """

        prefix = res_file.rstrip('0123456789')

        merged_res_file = self.merge(prefix)
        metrics = self.evaluator.eval(
            merged_res_file) if self.evaluator else {}
        return metrics

    def merge_eval_res(self, result_path):

        all_data = {
            "all": {
                "all_with_extra": {},
                "all_without_extra": {},

            },
            'noise': {'gaussian_noise': {}, 'shot_noise': {}, 'impulse_noise': {}},
            'blur': {'defocus_blur': {},
                     'glass_blur': {},
                     'motion_blur': {},
                     'zoom_blur': {}},
            'weather': {'snow': {}, 'frost': {}, 'fog': {}, 'brightness': {}},
            'digital': {'contrast': {},
                        'elastic_transform': {},
                        'pixelate': {},
                        'jpeg_compression': {}},
            'extra': {'speckle_noise': {},
                      'spatter': {},
                      'gaussian_blur': {},
                      'saturate': {}},

        }

        avg = []
        avg_wo_extra = []
        for noise in all_data:
            if noise == "all":
                continue
            for noise_type in all_data[noise]:
                lst = []
                for i in range(1, 6):
                    path = f"{result_path}/{noise}-{noise_type}-{i}-metric"
                    with open(path) as f:
                        lst.append(100 - float(json.load(f)['top1']))
                all_data[noise][noise_type] = np.average(lst)
                if noise != "extra":
                    avg_wo_extra.append(np.average(lst))
                avg.append(np.average(lst))
        all_data["all"]["all_with_extra"] = np.average(avg)
        all_data["all"]["all_without_extra"] = np.average(avg_wo_extra)

        table_st = self.get_table(all_data)
        self.logger.info("\n")
        for table in table_st:
            self.logger.info("\n" + str(table) + "\n")
        with open(f"{result_path}/robust.json", "w") as f:
            json.dump(all_data, f)

    @staticmethod
    def get_table(res):
        table_lst = []
        for key in res:

            table = PrettyTable()
            filed_list = []
            rows = []
            for subkey in res[key]:
                filed_list.append(f"{subkey}")
                rows.append(res[key][subkey])

            table.field_names = filed_list
            table.align = 'c'
            table.float_format = '.4'
            table.title = key
            table.add_row(list(rows))
            table_lst.append(table)
        return table_lst