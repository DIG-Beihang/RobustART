import os.path as osp
import numpy as np
from PIL import Image
import io
import cv2
import ffmpeg
import copy
import math
import random
import os
from tqdm import tqdm
try:
    import mc
except ImportError:
    pass
import argparse


pil_resize_mode_dict = {
    "pil-bilinear": Image.BILINEAR,
    "pil-nearest": Image.NEAREST,
    "pil-box": Image.BOX,
    "pil-hamming": Image.HAMMING,
    "pil-cubic": Image.BICUBIC,
    "pil-lanczos": Image.LANCZOS
}

cv_resize_mode_dict = {
    "opencv-nearest": cv2.INTER_NEAREST,
    "opencv-bilinear": cv2.INTER_LINEAR,
    "opencv-area": cv2.INTER_AREA,
    "opencv-cubic": cv2.INTER_CUBIC,
    "opencv-lanczos": cv2.INTER_LANCZOS4
}



class ImageTransfer:
    def __init__(self, root_dir=None, meta_file=None, save_root=None, decoder_type='pil',
                 resize_type='pil-bilinear', resize=224, transform_type='val', return_online=False, file_path=None):
        self.root_dir = root_dir
        self.meta_file = meta_file
        self.decoder_type = decoder_type
        self.resize_type = resize_type
        self.save_root = save_root
        self.initialized = False
        self.transform_type = transform_type
        self.return_online = return_online

        if isinstance(resize, tuple):
            self.resize = resize
        else:
            self.resize = (resize, resize)
        self.color_mode = 'RGB'

        if not self.return_online:
            new_meta_file_name = decoder_type + '_' + resize_type + '.txt'
            new_meta_file = open(new_meta_file_name, 'w')

            with open(meta_file) as f:
                lines = f.readlines()
            self.num = len(lines)
            self.metas = []
            for line in lines:
                filename, label = line.rstrip().split()
                self.metas.append({'filename': filename, 'label': label})

            save_dir = osp.join(save_root, decoder_type, resize_type)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)

            for idx in tqdm(range(self.num)):
                np_image = self.getimage(idx)
                save_file_name = self.metas[idx]['filename'] + '.npy'
                save_path = osp.join(save_dir, save_file_name)
                np.save(save_path, np_image)

                label = self.metas[idx]['label']
                new_meta_file.write(f'{osp.join(decoder_type, resize_type, save_file_name)} {label}'+'\n')

        else:
            self.file_path = file_path



    def getimage(self, idx=None):
        if not self.return_online:
            curr_meta = copy.deepcopy(self.metas[idx])
            filename = osp.join(self.root_dir, curr_meta['filename'])
            label = int(curr_meta['label'])
            # add root_dir to filename
            curr_meta['filename'] = filename
            img_bytes = self.read_file(curr_meta)
        else:
            img_bytes = self.read_file({'filename': self.file_path})
            filename = self.file_path
        img_after_decode = self.image_decoder(img_bytes, filepath=filename)
        assert isinstance(img_after_decode, np.ndarray)

        y, x, h, w = self.get_params(img_after_decode)
        img_after_resize = self.image_resize(img_after_decode, y, x, h, w)

        return img_after_resize


    def image_resize(self, img, y, x, h, w):
        if 'pil' in self.resize_type:
            img = self.toPIL(img)
            interpolation = pil_resize_mode_dict[self.resize_type]
        elif 'opencv' in self.resize_type:
            interpolation = cv_resize_mode_dict[self.resize_type]
        else:
            raise NotImplementedError

        if self.transform_type == 'train':
            i, j = y, x
            size = self.resize
            if 'pil' in self.resize_type:
                img = img.crop((j, i, j + w, i + h))
                return self.toNumpy(self.PIL_resize(img, size, interpolation))
            elif 'opencv' in self.resize_type:
                img = img[y: y + h, x: x + w]
                img = cv2.resize(img, self.resize, interpolation=interpolation)
                return img
            else:
                raise NotImplementedError
        elif self.transform_type == 'val':
            if 'pil' in self.resize_type:
                frist_resize = tuple(size * 8 / 7 for size in self.resize)
                img = self.PIL_resize(img, frist_resize, interpolation)

                w, h = img.size
                th, tw = self.resize
                i = int(round((h - th) / 2.))
                j = int(round((w - tw) / 2.))
                img = img.crop((j, i, j + tw, i + th))
                return self.toNumpy(img)
            elif 'opencv' in self.resize_type:
                width, height = tuple(int(size * 8 / 7) for size in self.resize)
                img = cv2.resize(img, (width, height), interpolation=interpolation)

                h, w, c = img.shape
                th, tw = self.resize
                dy = int(round((h - th) / 2.))
                dx = int(round((w - tw) / 2.))
                return img[dy: dy + th, dx: dx + tw]
        else:
            raise NotImplementedError




    def PIL_resize(self, img, size, interpolation):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)



    def toNumpy(self, img):
        return np.asarray(img)

    def toPIL(self, img):
        return Image.fromarray(img)

    def image_decoder(self, filebytes, filepath=None):
        if self.decoder_type == 'pil':
            buff = io.BytesIO(filebytes)
            try:
                with Image.open(buff) as img:
                    img = img.convert('RGB')
                    if self.color_mode == "BGR":
                        b, g, r = img.split()
                        img = Image.merge("RGB", (r, g, b))
                    elif self.color_mode == "GRAY":
                        img = img.convert('L')

            except IOError:
                print('Failed in loading {}'.format(filepath))
            image_array = np.array(img)
            return image_array
        elif self.decoder_type == 'opencv':
            try:
                img = cv2.imdecode(filebytes, cv2.IMREAD_COLOR)
                if self.color_mode == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif self.color_mode == "GRAY":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except IOError:
                print('Failed in loading {}'.format(filepath))
            return img
        elif self.decoder_type == 'ffmpeg':
            img = cv2.imdecode(filebytes, cv2.IMREAD_COLOR)
            height = img.shape[0]
            width = img.shape[1]
            out, _ = (
                     ffmpeg
                    .input(filepath)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True)
                 )
            img = (
                     np
                    .frombuffer(out, np.uint8)
                    .reshape([height, width, 3])
                 )
            return img
        else:
            raise NotImplementedError

    def get_params(self, img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """

        area = img.shape[0] * img.shape[1]
        height, width = img.shape[0], img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def read_file(self, meta_dict):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(meta_dict['filename'], value)
        value_str = mc.ConvertBuffer(value)
        filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)
        return filebytes

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument('--decoder', required=False, type=str, default='pil')
    parser.add_argument('--resize', required=False, type=str, default='pil-bilinear')
    parser.add_argument('--transform-type', required=False, type=str, default='val', choices=['val', 'train'])
    # train: Random Resize Crop
    # val: Resize (outsize * (8/7)) + Center Crop

    args = parser.parse_args()

    ImageTransfer(root_dir='/mnt/lustre/share/images/val', meta_file='/mnt/lustre/share/images/meta/val.txt',
                  save_root='/mnt/lustre/wangyan3/dataset-decoder-resize', decoder_type=args.decoder,
                  transform_type=args.transform_type, resize_type=args.resize)

