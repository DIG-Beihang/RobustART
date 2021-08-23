import json
import os.path as osp
from .base_dataset import BaseDataset
from prototype.prototype.data.image_reader import build_image_reader


class CustomDataset(BaseDataset):
    """
    Custom Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - osg_server (:obj:`str`): '10.198.3.28:30080/components/osg-default/v1'

    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG", "label": 0, "label_name": "dog"}\n"
    """
    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='mc', evaluator=None, image_reader_type='pil',
                 osg_server=None):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
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

        super(CustomDataset, self).__init__(root_dir=root_dir,
                                            meta_file=meta_file,
                                            read_from=read_from,
                                            transform=transform,
                                            evaluator=evaluator)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = osp.join(self.root_dir, curr_meta['filename'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        # 'label' and 'label_name' is optional for inference
        label = int(curr_meta['label']) if 'label' in curr_meta else 0
        label_name = curr_meta['label_name'] if 'label_name' in curr_meta else 'inference'

        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'label': label,
            'image_id': idx,
            'filename': filename,
            'label_name': label_name
        }
        return item

    def dump(self, writer, output):
        filename = output['filename']
        image_id = output['image_id']
        label_name = output['label_name']
        prediction = self.tensor2numpy(output['prediction'])
        score = self.tensor2numpy(output['score'])
        label = self.tensor2numpy(output['label'])
        for _idx in range(len(filename)):
            res = {
                'filename': filename[_idx],
                'image_id': int(image_id[_idx]),
                'label_name': label_name[_idx],
                'prediction': int(prediction[_idx]),
                'score': [float('%.8f' % s) for s in score[_idx]],
                'label': int(label[_idx])
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()
