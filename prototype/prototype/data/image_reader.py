import io
from PIL import Image
import logging

logger = logging.getLogger('global')


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img


def build_image_reader(reader_type):
    if reader_type == 'pil':
        return pil_loader
    else:
        raise NotImplementedError
