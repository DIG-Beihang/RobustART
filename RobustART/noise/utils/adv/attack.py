import foolbox as fb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch
import torch
import torch.nn as nn
from .Attacks.autoattack import AutoAttack
from .Attacks.imfgsm_attack import _mim_whitebox
from .Attacks.illc import ILLC
from .Attacks.pa import PA
from .Attacks.rllc import RLLC
from .Attacks.llc import LLC
from .Attacks.om import OM
from .Attacks.jsm import JSM
from .Attacks.ba import BA
from .Attacks.bim import BIM
from .Attacks.blb import BLB
from .Attacks.cw2 import CW2
from .Attacks.deepfool import DEEPFOOL
from .Attacks.ead import EAD
from .Attacks.rfgsm import RFGSM
from .Attacks.spsa import SPSA
from .Attacks.uap import UAP
from .Attacks.zoo import ZOO


def clip_l2_norm(cln_img, adv_img, eps):
    noise = adv_img - cln_img
    if torch.sqrt(torch.sum(noise**2)).item() > eps:
        clip_noise = noise * eps / torch.sqrt(torch.sum(noise**2))
        clip_adv = cln_img + clip_noise
        return clip_adv
    else:
        return adv_img


def pgd_linf(input, label, f_model, eps, rel_stepsize, steps):
    pgdlinf_att = fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=rel_stepsize, steps=steps)
    adv_fbpgd_linf, _, success = pgdlinf_att(f_model, input, label, epsilons=eps)
    return adv_fbpgd_linf

def pgd_l2(input, label, f_model, eps, rel_stepsize, steps):
    pgdl2_att = fb.attacks.L2ProjectedGradientDescentAttack(rel_stepsize=rel_stepsize, steps=steps)
    adv_fbpgd_l2, _, success = pgdl2_att(f_model, input, label, epsilons=eps)
    return adv_fbpgd_l2

def fgsm(input, label, f_model, eps):
    fgsm_att = fb.attacks.LinfFastGradientAttack()
    adv_fgsm, _, success = fgsm_att(f_model, input, label, epsilons=eps)
    return adv_fgsm

def autoattack_linf(input, label, model, norm, eps, version, verbose):
    aa_att = AutoAttack(model, norm=norm, eps=eps, version=version, verbose=verbose)
    adv_aa = aa_att.run_standard_evaluation(input, label, bs=input.shape[0])
    return adv_aa

def mim_linf(input, label, model, eps, num_steps, step_size, decay_factor):
    adv_mifgsm = _mim_whitebox(model, input, label, epsilon=eps, num_steps=num_steps, step_size=step_size, decay_factor=decay_factor)
    return adv_mifgsm

def pgd_l1(input, label, model, eps, input_size, eps_step, max_iter, batch_size):
    # using ART to gen PGD L1
    classifier = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), input_shape=(3, input_size, input_size), nb_classes=1000, clip_values=(0, 1), preprocessing=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), device_type='gpu')
    attack = ProjectedGradientDescentPyTorch(estimator=classifier, norm=1, eps=eps, eps_step=eps_step, max_iter=max_iter, num_random_init=1, batch_size=batch_size, verbose=False)
    adv_pgdl1 = attack.generate(x=input.cpu(), y=label.cpu())
    return torch.from_numpy(adv_pgdl1).cuda()


def illc(input, label, model, device, istarget, epsilon, epsilon_iter, num_steps):
    illc_att = ILLC(model=model, device=device, IsTargeted=istarget, epsilon=epsilon, epsilon_iter=epsilon_iter, num_steps=num_steps)
    adv_illc = illc_att.generate(xs=input, ys_target=label)
    return adv_illc.to(device)

def rllc(input, label, model, device, istarget, epsilon, alpha):
    rllc_att = RLLC(model=model, device=device, IsTargeted=istarget, epsilon=epsilon, alpha=alpha)
    adv_rllc = rllc_att.generate(xs=input, ys_target=label)
    return adv_rllc.to(device)

def pa(input, label, model, device, istarget, patch_path, position):
    pa_att = PA(model=model, device=device, IsTargeted=istarget, patch_path=patch_path, position=position)
    adv_pa = pa_att.generate(xs=input, ys=label)
    return adv_pa.to(device)

def ba(input, label, model, eps, delta, lower_bound, upper_bound, max_iter, binary_search_steps, batch_size, step_adapt, sample_size, init_size):
    attack = BA(model=model, eps=eps, delta=delta, lower_bound=lower_bound, upper_bound=upper_bound, max_iter=max_iter, binary_search_steps=binary_search_steps, batch_size=batch_size, step_adapt=step_adapt, sample_size=sample_size, init_size=init_size)
    adv_ba = attack.generate(xs=input, ys=label)
    return adv_ba

def bim(input, label, model, eps, eps_iter, num_steps):
    attack = BIM(model=model, eps=eps, eps_iter=eps_iter, num_steps=num_steps)
    adv_bim = attack.generate(xs=input, ys=label)
    return adv_bim

def blb(input, label, model, init_const, max_iter, binary_search_steps):
    attack = BLB(model=model, init_const=init_const, max_iter=max_iter, binary_search_steps=binary_search_steps)
    adv_blb = attack.generate(xs=input, ys=label)
    return adv_blb

def cw2(input, label, model, device, IsTarget, kappa, lr, init_const, lower_bound, upper_bound, max_iter, binary_search_steps):
    cw2_attack = CW2(model=model, device=device, IsTarget=IsTarget, kappa=kappa, lr=lr, init_const=init_const, lower_bound=lower_bound, upper_bound=upper_bound, max_iter=max_iter, binary_search_steps=binary_search_steps)
    adv_cw = cw2_attack.generate(input, label)
    return adv_cw

def deepfool(input, label, model, device, IsTarget, overshoot, max_iter):
    deepfool_attack = DEEPFOOL(model=model, device=device, IsTarget=IsTarget, overshoot=overshoot, max_iter=max_iter)
    adv_deepfool = deepfool_attack.generate(input, label)
    return adv_deepfool

def ead(input, label, model, device, IsTargeted, kappa, lr, init_const, lower_bound, upper_bound, max_iter, binary_search_steps, class_type_number, beta, EN):
    ead_attack = EAD(model=model, device=device, IsTargeted=IsTargeted, kappa=kappa, lr=lr, init_const=init_const, lower_bound=lower_bound, upper_bound=upper_bound, max_iter=max_iter, binary_search_steps=binary_search_steps, class_type_number=class_type_number, beta=beta, EN=EN)
    adv_ead = ead_attack.generate(input, label)
    return adv_ead

def llc(input, label, model, device, IsTargeted, epsilon):
    llc_att = LLC(model=model, device=device, IsTargeted=IsTargeted, epsilon=epsilon)
    adv_llc = llc_att.generate(xs=input, ys=label)
    return adv_llc

def om(input, label, model, device, IsTargeted, kappa, class_type_number, lr, init_const, lower_bound, upper_bound, max_iter, binary_search_steps, noise_count, noise_magnitude):
    att = OM(model=model, device=device, IsTargeted=IsTargeted, kappa=kappa, class_type_number=class_type_number, lr=lr, init_const=init_const, lower_bound=lower_bound, upper_bound=upper_bound, max_iter=max_iter, binary_search_steps=binary_search_steps, noise_count=noise_count, noise_magnitude=noise_magnitude)
    adv_om = att.generate(xs=input, ys=label)
    return adv_om

def jsm(input, label, model, device, IsTargeted, theta, gamma):
    att = JSM(model=model, device=device, IsTargeted=IsTargeted, theta=theta, gamma=gamma)
    adv_jsm = att.generate(xs=input, ys=label)
    return adv_jsm

def rfgsm(input,label,model, device, IsTargeted, eps, alp):
    att = RFGSM(model, device, IsTargeted, eps=eps, alp=alp)
    adv_xs=att.generate(xs=input,ys=label)
    return adv_xs

def spsa(input,label,model, device, IsTargeted, eps,learning_rate,delta,spsa_samples,spsa_iters,is_debug,sanity_checks,nb_iter):
    att=SPSA(model,device,IsTargeted,eps=eps,learning_rate=learning_rate,delta=delta,spsa_samples=spsa_samples,spsa_iters=spsa_iters,is_debug=is_debug,sanity_checks=sanity_checks,nb_iter=nb_iter)
    adv_xs=att.generate(xs=input,ys=label)
    return adv_xs

def uap(input,label,model, device, IsTargeted,dataset,deepfool_overshoot,deepfool_max_iter,fool_rate,uni_max_iter,epsilon):
    att=UAP(model, device, IsTargeted,dataset=dataset,deepfool_overshoot=deepfool_overshoot,deepfool_max_iter=deepfool_max_iter,fool_rate=fool_rate,uni_max_iter=uni_max_iter,epsilon=epsilon)
    adv_xs=att.generate(xs=input,ys=label)
    return adv_xs

def zoo(input,label,model, device, IsTargeted,solver,resize_init_size,img_h,img_w,num_channels,use_resize,class_type_number,use_tanh,confidence,batch_size,init_const,max_iter,binary_search_steps,beta1,beta2,lr,reset_adam_after_found,early_stop_iters,ABORT_EARLY,lower_bound,upper_bound,print_every,use_log,save_modifier,load_modifier,use_importance):
    att=ZOO(model, device, IsTargeted,solver=solver,resize_init_size=resize_init_size,img_h=img_h,img_w=img_w,num_channels=num_channels,use_resize=use_resize,class_type_number=class_type_number,use_tanh=use_tanh,confidence=confidence,batch_size=batch_size,init_const=init_const,max_iter=max_iter,binary_search_steps=binary_search_steps,beta1=beta1,beta2=beta2,lr=lr,reset_adam_after_found=reset_adam_after_found,early_stop_iters=early_stop_iters,ABORT_EARLY=ABORT_EARLY,lower_bound=lower_bound,upper_bound=upper_bound,print_every=print_every,use_log=use_log,save_modifier=save_modifier,load_modifier=load_modifier,use_importance=use_importance)
    adv_xs=att.generate(xs=input,ys=label)
    return adv_xs

attack_list = {'pgd_l1': pgd_l1, 'pgd_linf': pgd_linf, 'pgd_l2': pgd_l2, 'fgsm': fgsm, 'autoattack_linf': autoattack_linf, 'mim_linf': mim_linf, 'cw2': cw2, 'deepfool': deepfool, 'ead': ead, 'ba': ba, 'bim': bim, 'blb': blb, 'llc': llc, 'om': om, 'jsm': jsm, 'illc': illc, 'rllc': rllc, 'pa': pa,'rfgsm':rfgsm,'spsa':spsa,'uap':uap,'zoo':zoo}

