import json
import os.path as osp
from .base_dataset import BaseDataset
from prototype.prototype.data.image_reader import build_image_reader


class MultiClassDataset(BaseDataset):
    """
    Dataset that supports multi-class classification.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - osg_server (:obj:`str`): '10.198.3.28:30080/components/osg-default/v1'

    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG",
          "label_list": [0, 1, 1],
          "label_name_list": ["hat", "glass", "jacket"]}\n"
    """
    def __init__(self, root_dir, meta_file, transform=None, read_from='mc',
                 evaluator=None, image_reader_type='pil', osg_server=None):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.read_from = read_from
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.osg_server = osg_server
        self.initialized = False

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            info = json.loads(line)
            self.metas.append(info)

        super(MultiClassDataset, self).__init__(root_dir=root_dir, meta_file=meta_file,
                                                read_from=read_from, transform=transform, evaluator=evaluator)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = osp.join(self.root_dir, curr_meta['filename'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        # 'label_list' and 'label_name_list' is optional for inference
        label_list = curr_meta['label_list'] if 'label_list' in curr_meta else None
        label_name_list = curr_meta['label_name_list'] if 'label_name_list' in curr_meta else None

        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'label_list': label_list,
            'label_name_list': label_name_list,
            'filename': filename,
            'image_id': idx
        }
        return item

    def dump(self, writer, output):
        filename = output['filename']
        image_id = output['image_id']
        label_name_list = output['label_name_list']
        prediction = self.tensor2numpy(output['prediction'])
        score = self.tensor2numpy(output['score'])
        label_list = self.tensor2numpy(output['label_list'])
        for _idx in range(len(filename)):
            _label_name_list = []
            _label_list = []
            _prediction = []
            _score = []
            for _att_idx in range(len(label_name_list)):
                _label_name_list.append(label_name_list[_att_idx][0])
                _label_list.append(int(label_list[_att_idx][_idx]))
                _prediction.append(int(prediction[_att_idx][_idx]))
                _score.append([float('%.8f' % s) for s in score[_att_idx][_idx]])
            res = {
                'filename': filename[_idx],
                'image_id': int(image_id[_idx]),
                'label_name_list': _label_name_list,
                'prediction': _prediction,
                'score': _score,
                'label_list': _label_list,
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()
