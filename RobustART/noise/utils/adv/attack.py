import foolbox as fb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch
import torch
import torch.nn as nn
from .Attacks.autoattack import AutoAttack
from .Attacks.imfgsm_attack import _mim_whitebox


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


attack_list = {'pgd_l1': pgd_l1, 'pgd_linf': pgd_linf, 'pgd_l2': pgd_l2, 'fgsm': fgsm, 'autoattack_linf': autoattack_linf, 'mim_linf': mim_linf}
