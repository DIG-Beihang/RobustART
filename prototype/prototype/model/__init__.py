from .mobilenet_v2 import mobilenet_v2  # noqa: F401
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50,
    resnet101, resnet152, resnet_custom
)
from .preact_resnet import (  # noqa: F401
    preact_resnet18, preact_resnet34, preact_resnet50,
    preact_resnet101, preact_resnet152
)
from .efficientnet import (  # noqa: F401
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_b0_nodrop, efficientnet_b1_nodrop, efficientnet_b2_nodrop, efficientnet_b3_nodrop,
    efficientnet_b4_nodrop, efficientnet_b5_nodrop, efficientnet_b6_nodrop, efficientnet_b7_nodrop
)
from .shufflenet_v2 import (  # noqa: F401
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, shufflenet_v2_scale
)
from .senet import se_resnext50_32x4d, se_resnext101_32x4d  # noqa: F401
from .densenet import densenet121, densenet169, densenet201, densenet161  # noqa: F401
# from .toponet import toponet_conv, toponet_sepconv, toponet_mb
from .hrnet import HRNet  # noqa: F401
from .mnasnet import mnasnet  # noqa: F401
from .nas_zoo import (  # noqa: F401
    mbnas_t29_x0_84, mbnas_t47_x1_00, supnas_t18_x1_00, supnas_t37_x0_92, supnas_t44_x1_00,
    supnas_t66_x1_11, supnas_t100_x0_96, nas_custom
)
from .resnet_official import (  # noqa: F401
    resnet18_official, resnet34_official, resnet50_official, resnet101_official, resnet152_official,
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
)
from .condconv_resnet import (  # noqa: F401
    resnet18_condconv_shared, resnet18_condconv_independent,
    resnet34_condconv_shared, resnet34_condconv_independent,
    resnet50_condconv_shared, resnet50_condconv_independent,
    resnet101_condconv_shared, resnet101_condconv_independent,
    resnet152_condconv_shared, resnet152_condconv_independent
)
from .condconv_mobilenet_v2 import (  # noqa: F401
    mobilenetv2_condconv_pointwise, mobilenetv2_condconv_independent, mobilenetv2_condconv_shared
)
from .mobilenet_v3 import mobilenet_v3  # noqa: F401
from .ghostnet import ghostnet  # noqa: F401
from .resnest import resnest50, resnest101, resnest200, resnest269  # noqa: F401
from .ibnnet import resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a  # noqa: F401
from .weightnet_shufflenet_v2 import (  # noqa: F401
    shufflenet_v2_x0_5_weightnet, shufflenet_v2_x1_0_weightnet,
    shufflenet_v2_x1_5_weightnet, shufflenet_v2_x2_0_weightnet
)
from .mobilenext import mobilenext  # noqa: F401
from .dmcp_resnet import (  # noqa: F401
    dmcp_resnet18_45M, dmcp_resnet18_47M, dmcp_resnet18_51M, dmcp_resnet18_480M, dmcp_resnet50_282M,
    dmcp_resnet50_1100M, dmcp_resnet50_2200M
)
from .bignas_resnet_basicblock import (  # noqa: F401
    bignas_resnet18_9M, bignas_resnet18_37M, bignas_resnet18_50M,
    bignas_resnet18_49M, bignas_resnet18_66M, bignas_resnet18_1555M,
    bignas_resnet18_107M, bignas_resnet18_125M, bignas_resnet18_150M,
    bignas_resnet18_312M, bignas_resnet18_403M, bignas_resnet18_492M
)
from .bignas_resnet_bottleneck import (  # noqa: F401
    bignas_resnet50_2954M, bignas_resnet50_3145M, bignas_resnet50_3811M
)
from .sparse_resnet import (  # noqa: F401
    sparse_resnet18, sparse_resnet26, sparse_resnet50, sparse_resnet101, sparse_resnet152
)
from .vision_transformer import (  # noqa: F401
    vit_b32_224, vit_b16_224, hybird_vit_res50_b16_224,
    deit_tiny_b16_224, deit_small_b16_224, deit_base_b16_224
)
from .nfnet import (  # noqa: F401
    nfnet_F0, nfnet_F1, nfnet_F2,
    nfnet_F3, nfnet_F4, nfnet_F5, nfnet_F6
)
# add from .vit
from .vit.swin_transformer import (  # noqa: F401
    swin_tiny, swin_small, swin_base_224, swin_base_384, swin_large_224, swin_large_384
)
from .vit.mlp_mixer import (  # noqa: F401
    mixer_b16_224, mixer_L16_224
)
from .vit.vit_base import (  # noqa: F401
    new_deit_small_patch16_224
)
from .vit.levit import (  # noqa: F401
    LeViT_128S, LeViT_128, LeViT_192, LeViT_256, LeViT_384
)
from .repvgg import (  # noqa: F401
    repvgg_A0, repvgg_A1, repvgg_A2, repvgg_B0,
    repvgg_B1, repvgg_B1g2, repvgg_B1g4,
    repvgg_B2, repvgg_B2g2, repvgg_B2g4,
    repvgg_B3, repvgg_B3g2, repvgg_B3g4
)

from .tinynet import (  # noqa: F401
    tinynet_a, tinynet_b, tinynet_c, tinynet_d, tinynet_e
)
from .alexnet import alexnet


def get_model_robust_baseline():
    return {
        '21k_resnet50': resnet50_official(),
        'resnet50_augmix': resnet50_official(),
        'resnet50_mococv2': resnet50_official(),
        'resnet50_adamw': resnet50_official(),
        'regnetx_3200m_augmix': regnetx_3200m(),
        'regnetx_3200m_adamw': regnetx_3200m(),
        'regnetx_6400m': regnetx_6400m(),
        'repvgg_A0_deploy': repvgg_A0(),
        'repvgg_B3_deploy': repvgg_B3(),
        "shufflenet_v2_x0_5": shufflenet_v2_x0_5(),
        "shufflenet_v2_x1_5": shufflenet_v2_x1_5(),
        'shufflenet_v2_x2_0_augmentation': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_augmix': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_ema': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_label_smooth': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_adamw': shufflenet_v2_x2_0(),
        "efficientnet_b0": efficientnet_b0(),
        'efficientnet_b0_nodrop': efficientnet_b0_nodrop(),
        'efficientnet_b1_nodrop_240': efficientnet_b1_nodrop(),
        'efficientnet_b2_nodrop_260': efficientnet_b2_nodrop(),
        'efficientnet_b3_nodrop_300': efficientnet_b3_nodrop(),
        'efficientnet_b4_nodrop_380': efficientnet_b4_nodrop(),
        "mobilenet_v3_large_x1_4": mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_augmentation': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_augmix': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_ema': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_label_smooth': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adv_train': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adamw': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_dropout': mobilenet_v3(scale=1.4, dropout=0.2, mode='large'),
        '21k_vit_base_patch16_224': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768),
        'vit_base_patch16_224_withdrop': vit_b16_224(drop_path=0.1, qkv_bias=True,
                                   representation_size=768),
        'mixer_B16_224_augmentation':mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_augmix': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_ema': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_label_smooth.pth.tar': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_adv_train': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_withdrop': mixer_b16_224(drop_path=0.1, drop_path_rate=0.1),
        'mixer_L16_224': mixer_L16_224(drop_path=0.0, drop_path_rate=0.0),

    }

"""    return {
        'alexnet': alexnet(dropout=0.0),
        "deit_base_b16_224": deit_base_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True),
        "deit_small_b16_224": deit_small_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True),
        "deit_tiny_b16_224": deit_tiny_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True),
        "densenet121": densenet121(),
        "densenet169": densenet169(),
        "densenet201": densenet201(),
        "efficientnet_b0": efficientnet_b0(),
        "efficientnet_b1": efficientnet_b0(),
        "efficientnet_b2": efficientnet_b0(),
        "efficientnet_b3": efficientnet_b0(),
        "efficientnet_b4": efficientnet_b0(),
        "efficientnet_b5": efficientnet_b0(),
        "efficientnet_b6": efficientnet_b0(),
        "efficientnet_b7": efficientnet_b0(),
        "mixer_b16_224": mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        "mixer_L16_224": mixer_L16_224(drop_path=0.0, drop_path_rate=0.0),
        "mobilenet_v2_x0_5": mobilenet_v2(scale=0.5),
        "mobilenet_v2_x0_75": mobilenet_v2(scale=0.75),
        "mobilenet_v2_x1_0": mobilenet_v2(scale=1.0),
        "mobilenet_v2_x1_4": mobilenet_v2(scale=1.4),
        "mobilenet_v3_large_x0_5": mobilenet_v3(scale=0.5, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x0_35": mobilenet_v3(scale=0.35, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x0_75": mobilenet_v3(scale=0.75, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x1_0": mobilenet_v3(scale=1.0, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x1_4": mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        "regnetx_400m": regnetx_400m(),
        "regnetx_800m": regnetx_800m(),
        "regnetx_1600m": regnetx_1600m(),
        "regnetx_3200m": regnetx_3200m(),
        "regnetx_6400m": regnetx_6400m(),
        "repvgg_A0": repvgg_A0(),
        "repvgg_B3": repvgg_B3(),
        "resnet18": resnet18_official(),
        "resnet34": resnet34_official(),
        "resnet50": resnet50_official(),
        "resnet101": resnet101_official(),
        "resnet152": resnet152_official(),
        "resnext50_32x4d": resnext50_32x4d(),
        "resnext101_32x8d": resnext101_32x8d(),
        "shufflenet_v2_x0_5": shufflenet_v2_x0_5(),
        "shufflenet_v2_x1_0": shufflenet_v2_x1_0(),
        "shufflenet_v2_x1_5": shufflenet_v2_x1_5(),
        "shufflenet_v2_x2_0": shufflenet_v2_x2_0(),
        "vit_b16_224": vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768),
        "vit_b32_224": vit_b32_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768),
        "wide_resnet50_2": wide_resnet50_2(),
        "wide_resnet101_2": wide_resnet101_2()

    }"""

def get_c():
    return {
        "efficientnet_b0": efficientnet_b0(),
        'efficientnet_b0_nodrop': efficientnet_b0_nodrop(),
        'efficientnet_b1_nodrop_240': efficientnet_b1_nodrop(),
        'efficientnet_b2_nodrop_260': efficientnet_b2_nodrop(),
        'efficientnet_b3_nodrop_300': efficientnet_b3_nodrop(),
        'efficientnet_b4_nodrop_380': efficientnet_b4_nodrop(),
        'shufflenet_v2_x2_0_adv_train': shufflenet_v2_x2_0(),
        'mixer_b16_224_withdrop': mixer_b16_224(drop_path=0.1, drop_path_rate=0.1),
        'mixer_B16_224_adv_train': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
    }

def get_s():
    return {
        '21k_resnet50': resnet50_official(),
        'resnet50_augmix': resnet50_official(),
        'resnet50_mococv2': resnet50_official(),
        'resnet50_adamw': resnet50_official(),
        'regnetx_3200m_augmix': regnetx_3200m(),
        'regnetx_3200m_adamw': regnetx_3200m(),
        'repvgg_A0_deploy': repvgg_A0(),
        'repvgg_B3_deploy': repvgg_B3(),
        'shufflenet_v2_x2_0_augmentation': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_augmix': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_ema': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_label_smooth': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_adamw': shufflenet_v2_x2_0(),
        "efficientnet_b0": efficientnet_b0(),
        'efficientnet_b0_nodrop': efficientnet_b0_nodrop(),
        'efficientnet_b1_nodrop_240': efficientnet_b1_nodrop(),
        'efficientnet_b2_nodrop_260': efficientnet_b2_nodrop(),
        'efficientnet_b3_nodrop_300': efficientnet_b3_nodrop(),
        'efficientnet_b4_nodrop_380': efficientnet_b4_nodrop(),
        'mobilenet_v3_large_x1_4_augmix': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adamw': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_dropout': mobilenet_v3(scale=1.4, dropout=0.2, mode='large'),
        '21k_vit_base_patch16_224': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768),
        'vit_base_patch16_224_withdrop': vit_b16_224(drop_path=0.1, qkv_bias=True,
                                   representation_size=768),
        'mixer_b16_224_augmix': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_adv_train': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_withdrop': mixer_b16_224(drop_path=0.1, drop_path_rate=0.1),
        'mixer_L16_224': mixer_L16_224(drop_path=0.0, drop_path_rate=0.0),
    }

def get_gau():
    return {
        '21k_resnet50': resnet50_official(),
        'resnet50_augmix': resnet50_official(),
        'resnet50_mococv2': resnet50_official(),
        'resnet50_adamw': resnet50_official(),
        'regnetx_3200m_augmix': regnetx_3200m(),
        'regnetx_3200m_adamw': regnetx_3200m(),
        'regnetx_6400m': regnetx_6400m(),
        'repvgg_A0_deploy': repvgg_A0(),
        'repvgg_B3_deploy': repvgg_B3(),
        'shufflenet_v2_x2_0_augmentation': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_augmix': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_ema': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_label_smooth': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_adamw': shufflenet_v2_x2_0(),
        "efficientnet_b0": efficientnet_b0(),
        'efficientnet_b0_nodrop': efficientnet_b0_nodrop(),
        'efficientnet_b1_nodrop_240': efficientnet_b1_nodrop(),
        'efficientnet_b2_nodrop_260': efficientnet_b2_nodrop(),
        'efficientnet_b3_nodrop_300': efficientnet_b3_nodrop(),
        'efficientnet_b4_nodrop_380': efficientnet_b4_nodrop(),
        'shufflenet_v2_x2_0_adv_train': shufflenet_v2_x2_0(),
        'mobilenet_v3_large_x1_4_augmentation': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_augmix': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_ema': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_label_smooth': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adv_train': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adamw': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_dropout': mobilenet_v3(scale=1.4, dropout=0.2, mode='large'),
        '21k_vit_base_patch16_224': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                                representation_size=768),
        'vit_base_patch16_224_withdrop': vit_b16_224(drop_path=0.1, qkv_bias=True,
                                                     representation_size=768),
        'mixer_B16_224_augmentation': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_augmix': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_ema': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_label_smooth.pth.tar': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_adv_train': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_withdrop': mixer_b16_224(drop_path=0.1, drop_path_rate=0.1),

    }

def get_p():
    return {
        'mobilenet_v3_large_x1_4_augmix': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adamw': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_dropout': mobilenet_v3(scale=1.4, dropout=0.2, mode='large'),
        '21k_vit_base_patch16_224': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                                representation_size=768),
        'vit_base_patch16_224_withdrop': vit_b16_224(drop_path=0.1, qkv_bias=True,
                                                     representation_size=768),
        'mixer_b16_224_augmix': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_adv_train': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_withdrop': mixer_b16_224(drop_path=0.1, drop_path_rate=0.1),
    }

def get_efficient():
    return {
        "efficientnet_b0": efficientnet_b0(),
        'efficientnet_b0_nodrop': efficientnet_b0_nodrop(),
        'efficientnet_b1_nodrop_240': efficientnet_b1_nodrop(),
        'efficientnet_b2_nodrop_260': efficientnet_b2_nodrop(),
        'efficientnet_b3_nodrop_300': efficientnet_b3_nodrop(),
        'efficientnet_b4_nodrop_380': efficientnet_b4_nodrop(),
    }


def get_model_robust_trick():
    return {
        'mixer_B16_224_augmentation': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_ema': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_label_smooth': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mobilenet_v3_large_x1_4_adv_train': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_augmentation': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_ema': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_label_smooth': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'regnetx3200m_adv_train': regnetx_3200m(),
        'regnetx3200m_augmentation': regnetx_3200m(),
        'regnetx3200m_ema': regnetx_3200m(),
        'regnetx3200m_label_smooth': regnetx_3200m(),
        'resnet50_adv_train': resnet50_official(),
        'resnet50_augmentation': resnet50_official(),
        'resnet50_ema': resnet50_official(),
        'resnet50_label_smooth': resnet50_official(),
        'shufflenet_v2_x2_0_adv_train': shufflenet_v2_x2_0(),
        'vit_base_patch16_224_augmentation': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                                         representation_size=768),
        'vit_base_patch16_224_ema': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                                representation_size=768),
        'vit_base_patch16_224_label_smooth': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                                         representation_size=768)
    }

def model_entry(config):
    if config['type'] not in globals():
        if config['type'].startswith('spring_'):
            try:
                from spring.models import SPRING_MODELS_REGISTRY
            except ImportError:
                print('Please install Spring2 first!')
            model_name = config['type'][len('spring_'):]
            config['type'] = model_name
            return SPRING_MODELS_REGISTRY.build(config)

    return globals()[config['type']](**config['kwargs'])


get_model = model_entry
