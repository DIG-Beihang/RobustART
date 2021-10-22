from PIL import Image
from RobustART.noise.utils.imagenet_c import corrupt
from RobustART.noise.utils.adv import pgd_l1, pgd_l2, pgd_linf, clip_l2_norm, autoattack_linf, mim_linf, fgsm, rfgsm,spsa_spsa,spsa,uap,zoo
from .imagenet_s_gen import ImageTransfer
import numpy as np


noise_list = ['imagenet-s', 'imagenet-c', 'pgd_linf', 'pgd_l2', 'fgsm', 'autoattack_linf', 'mim_linf', 'pgd_l1','rfgsm','spsa_clip_eta','spsa_spsa','spsa','uap','zoo']

default_config = {
    'imagenet-s': {'decoder_type': 'pil', 'resize_type': 'pil-bilinear', 'transform_type': 'val'},
    'imagenet-c': {'severity': 1, 'corruption_name': None, 'corruption_number': -1}, 
    'pgd_linf': {'f_model': None, 'eps': 8/255, 'rel_stepsize': 3/40, 'steps': 20},
    'pgd_l2': {'f_model': None, 'eps': 8.0, 'rel_stepsize': 3/40, 'steps': 20},
    'fgsm': {'f_model': None, 'eps': 8/255},
    'autoattack_linf': {'model': None, 'norm': 'Linf', 'eps': 8/255, 'version': 'standard', 'verbose': False},
    'mim_linf': {'model': None, 'eps': 8/255, 'num_steps': 20, 'step_size': 0.002, 'decay_factor': 1.0},
    'pgd_l1': {'model': None, 'eps': 1600.0, 'input_size': 224, 'eps_step': 120, 'max_iter': 20, 'batch_size': 16},

    'rfgsm':{'model':None,'device':None,'IsTargeted':None,'eps':0.1,'alp':0.5,'xs':None,'ys':None},
    'spsa_spsa':{'model':None,'device':None,'IsTargeted':None,'eps':0.05,'learning_rate':0.01,'delta':0.01,'spsa_samples':32,'spsa_iters':2,'is_debug':False,'sanity_checks':True,'nb_iter':20,'x':None,'y':None,'norm':np.inf,'clip_min':-np.inf,'clip_max':np.inf,'early_stop_loss_threshold':None},
    'spsa':{'model':None,'device':None,'IsTargeted':None,'eps':0.05,'learning_rate':0.01,'delta':0.01,'spsa_samples':32,'spsa_iters':2,'is_debug':False,'sanity_checks':True,'nb_iter':20,'xs':None,'ys':None},
    'uap':{'model':None,'device':None,'IsTargeted':None,'dataset':'cifar10','deepfool_overshoot':0.02,'deepfool_max_iter':50,'fool_rate':0.5,'uni_max_iter':100,'epsilon':0.1},
    'zoo':{'model':None,'device':None,'IsTargeted':None,'solver':'Newton','resize_init_size':32,'img_h':224,'img_w':224,'num_channels':3,'use_resize':False,'class_type_number':10,'use_tanh':True,'confidence':0,'batch_size':32,'init_const':5,'max_iter':100,'binary_search_steps':1,'beta1':0.9,'beta2':0.999,'lr':1e-2,'reset_adam_after_found':False,'early_stop_iters':30,'ABORT_EARLY':True,'lower_bound':'0.0','upper_bound':1.0,'print_every':10,'use_log':True,'save_modifier':'','load_modifier':'','use_importance':False}
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
    'mim_linf': mim_linf,
    'rfgsm':rfgsm,
    'spsa_spsa':spsa_spsa,
    'spsa':spsa,
    'uap':uap,
    'zoo':zoo
}

