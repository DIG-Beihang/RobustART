from PIL import Image
from RobustART.noise.utils.imagenet_c import corrupt
from RobustART.noise.utils.adv import pgd_l1, pgd_l2, pgd_linf, clip_l2_norm, autoattack_linf, mim_linf, fgsm
from .imagenet_s_gen import ImageTransfer


noise_list = ['imagenet-s', 'imagenet-c', 'pgd_linf', 'pgd_l2', 'fgsm', 'autoattack_linf', 'mim_linf', 'pgd_l1']

default_config = {
    'imagenet-s': {'decoder_type': 'pil', 'resize_type': 'pil-bilinear', 'transform_type': 'val'},
    'imagenet-c': {'severity': 1, 'corruption_name': None, 'corruption_number': -1}, 
    'pgd_linf': {'f_model': None, 'eps': 8/255, 'rel_stepsize': 3/40, 'steps': 20},
    'pgd_l2': {'f_model': None, 'eps': 8.0, 'rel_stepsize': 3/40, 'steps': 20},
    'fgsm': {'f_model': None, 'eps': 8/255},
    'autoattack_linf': {'model': None, 'norm': 'Linf', 'eps': 8/255, 'version': 'standard', 'verbose': False},
    'mim_linf': {'model': None, 'eps': 8/255, 'num_steps': 20, 'step_size': 0.002, 'decay_factor': 1.0},
    'pgd_l1': {'model': None, 'eps': 1600.0, 'input_size': 224, 'eps_step': 120, 'max_iter': 20, 'batch_size': 16}
}



def add_noise_for_imagenet_c(image, severity=1, corruption_name=None, corruption_number=-1):
    if isinstance(image, str):
        img = Image.open(image, 'r')
        return corrupt(img, severity=severity, corruption_name=corruption_name, corruption_number=corruption_number)
    else:
        b = image.shape[0]
        for i in range(b):
            img = Image.fromarray(image[i])
            image[i] = corrupt(img, severity=severity, corruption_name=corruption_name, corruption_number=corruption_number)
        return image


def add_noise_for_imagenet_s(image, decoder_type='pil', resize_type='pil-bilinear', transform_type='val'):
    assert isinstance(image, str), "Input of imagenet-S can only be file path"
    imagS = ImageTransfer(file_path=image, decoder_type=decoder_type, resize_type=resize_type,
                          transform_type=transform_type, return_online=True)
    return imagS.getimage()


function_dict = {
    'imagenet-s': add_noise_for_imagenet_s,
    'imagenet-c': add_noise_for_imagenet_c,
    'pgd_l1': pgd_l1,
    'pgd_linf': pgd_linf,
    'pgd_l2': pgd_l2,
    'fgsm': fgsm,
    'autoattack_linf': autoattack_linf,
    'mim_linf': mim_linf
}

