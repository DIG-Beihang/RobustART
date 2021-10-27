from PIL import Image
from RobustART.noise.utils.imagenet_c import corrupt
from RobustART.noise.utils.adv import pgd_l1, pgd_l2, pgd_linf, clip_l2_norm, autoattack_linf, mim_linf, fgsm, cw2, deepfool, illc, rllc, pa, ead, ba, bim, blb, llc, jsm, om, fgsm, rfgsm,spsa,uap,zoo
from .imagenet_s_gen import ImageTransfer


noise_list = ['imagenet-s', 'imagenet-c', 'pgd_linf', 'pgd_l2', 'fgsm', 'autoattack_linf', 'mim_linf', 'pgd_l1', 'cw2', 'deepfool', 'illc', 'rllc', 'pa', 'ead', 'llc', 'om', 'jsm','rfgsm','spsa','uap','zoo']


default_config = {
    'imagenet-s': {'decoder_type': 'pil', 'resize_type': 'pil-bilinear', 'transform_type': 'val'},
    'imagenet-c': {'severity': 1, 'corruption_name': None, 'corruption_number': -1}, 
    'pgd_linf': {'f_model': None, 'eps': 8/255, 'rel_stepsize': 3/40, 'steps': 20},
    'pgd_l2': {'f_model': None, 'eps': 8.0, 'rel_stepsize': 3/40, 'steps': 20},
    'fgsm': {'f_model': None, 'eps': 8/255},
    'autoattack_linf': {'model': None, 'norm': 'Linf', 'eps': 8/255, 'version': 'standard', 'verbose': False},
    'mim_linf': {'model': None, 'eps': 8/255, 'num_steps': 20, 'step_size': 0.002, 'decay_factor': 1.0},
    'pgd_l1': {'model': None, 'eps': 1600.0, 'input_size': 224, 'eps_step': 120, 'max_iter': 20, 'batch_size': 16},
    'llc': {'model': None, 'device': None, 'IsTargeted': None, 'epsilon': 0.01},
    'jsm': {'model': None, 'device': None, 'IsTargeted': None, 'theta': 1.0, 'gamma': 0.001},
    'om': {'model': None, 'device': None, 'IsTargeted': None, 'kappa': 0, 'class_type_number': 1000, 'lr': 0.2, 'init_const': 0.02, 'lower_bound': 0.0, 'upper_bound': 1.0, 'max_iter': 5, 'binary_search_steps': 3, 'noise_count': 20, 'noise_magnitude': 0.3},
    'ba': {'eps': 0.01, 'delta': 0.01, 'lower_bound': 0.0, 'upper_bound': 1.0, 'max_iter': 10, 'binary_search_steps': 20, 'batch_size': 8, 'step_adapt': 0.9, 'sample_size': 80, 'init_size': 200},
    'bim': {'eps': 0.1, 'eps_iter': 0.1, 'num_steps': 15},
    'blb': {'init_const': 0.01, 'max_iter': 1000, 'binary_search_steps': 5},
    'cw2': {'model': None, 'device': None, 'IsTarget': None, 'kappa': 0, 'lr': 0.2, 'init_const': 0.01, 'lower_bound': 0.0, 'upper_bound': 1.0, 'max_iter': 200, 'binary_search_steps': 4},
    'deepfool': {'model': None, 'device': None, 'IsTarget': None, 'overshoot': 0.02, 'max_iter': 10},
    'ead': {'model': None, 'device': None, 'IsTarget': None, 'kappa': 0, 'lr': 0.2, 'init_const': 0.02, 'lower_bound': 0.0, 'upper_bound': 1.0, 'max_iter': 50, 'binary_search_steps': 3, 'class_type_number': 1000, 'beta': 1e-3, 'EN': True},
    'illc': {'model': None, 'device': None, 'istarget': None, 'epsilon': 0.3, 'epsilon_iter': 0.5, 'num_steps': 10},
    'rllc': {'model': None, 'device': None, 'istarget': None, 'epsilon': 0.1, 'alpha': 0.4},
    'pa': {'model': None, 'device': None, 'istarget': None, 'patch_path': '', 'position': "128,128"},
    'rfgsm':{'model':None,'device':None,'IsTargeted':None,'eps':0.1,'alp':0.5},
    'spsa':{'model':None,'device':None,'IsTargeted':None,'eps':0.05,'learning_rate':0.01,'delta':0.01,'spsa_samples':32,'spsa_iters':2,'is_debug':False,'sanity_checks':True,'nb_iter':20},
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
    'jsm': jsm,
    'llc': llc,
    'om': om,
    'ba': ba,
    'bim': bim,
    'blb': blb,
    'cw2': cw2,
    'deepfool': deepfool,
    'ead': ead,
    'illc': illc,
    'rllc': rllc,
    'pa': pa,
    'rfgsm':rfgsm,
    'spsa':spsa,
    'uap':uap,
    'zoo':zoo
}

