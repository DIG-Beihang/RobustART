import torch.nn as nn
from .vit_base import VisionTransformer
from .layers import get_sinusoid_encoding, T2T_module
from .default_cfg import default_kwargs


default_kwargs.update({
    'embed_dim': 384,
    'depth': 14,
    'mlp_ratio': 3.,
    'num_heads': 6,
    'qkv_bias': False
})
default_kwargs.pop('patch_size')
default_kwargs.pop('representation_size')


class T2T_ViT(VisionTransformer):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., token_dim=64):

        self.tokens_type = tokens_type
        self.token_dim = token_dim

        patch_size = None
        representation_size = None
        super(T2T_ViT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                      num_heads, mlp_ratio, qkv_bias, qk_scale, representation_size,
                                      drop_rate, attn_drop_rate, drop_path_rate)

    # override
    def init_building_blocks(self):
        super(T2T_ViT, self).init_building_blocks()
        self.embed_layer = T2T_module

    # override
    def init_patch_emb(self):
        self.tokens_to_token = T2T_module(
                img_size=self.img_size, tokens_type=self.tokens_type, in_chans=self.in_chans,
                embed_dim=self.embed_dim, token_dim=self.token_dim)
        self.patch_embed = self.tokens_to_token

    # override
    def init_pos_emb(self, num_patches, drop_rate):
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=self.embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)


def T2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    # default_kwargs.update({'tokens_type': 'transformer',
    #                         'qk_scale': 384 ** -0.5})
    default_kwargs.update({'tokens_type': 'transformer'})
    default_kwargs.update(kwargs)
    model = T2T_ViT(**default_kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def T2t_vit_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    default_kwargs.update({'tokens_type': 'performer',
                           'qk_scale': 384 ** -0.5})
    default_kwargs.update(kwargs)
    model = T2T_ViT(**default_kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

