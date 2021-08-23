import torch
import torch.nn as nn
from .layers import VanillaPatchEmbed, VanillaAttention, VanillaMlp, DropPath
from functools import partial
from collections import OrderedDict
from .default_cfg import default_kwargs as _cfg
from prototype.prototype.utils.trunc_normal_initializer import trunc_normal_


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VanillaAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = VanillaMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., apply_layer_norm=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            apply_layer_norm (bool): indicator of whether to add norm_layer to patch_embed
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.depth = depth

        # initialize building blocks of vit
        self.init_building_blocks()

        # patch embedding
        self.init_patch_emb(apply_layer_norm)

        # class token
        self.init_cls_token(embed_dim)

        # position embedding
        self.init_pos_emb(self.patch_embed.num_patches, drop_rate)

        # initialize transformer
        self.init_transformer()

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_param_no_wd(self):
        no_wd = [self.pos_embed, self.cls_token]
        return no_wd

    def init_embed_layer(self):
        self.embed_layer = VanillaPatchEmbed

    def init_norm_layer(self):
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

    def init_act_layer(self):
        self.act_layer = nn.GELU

    def init_block_type(self):
        self.block_type = Block

    def init_building_blocks(self):
        self.init_embed_layer()
        self.init_norm_layer()
        self.init_act_layer()
        self.init_block_type()

    def init_patch_emb(self, apply_layer_norm=False):
        apply_layer_norm = self.norm_layer if apply_layer_norm else None
        self.patch_embed = self.embed_layer(img_size=self.img_size, patch_size=self.patch_size,
                                            in_chans=self.in_chans, embed_dim=self.embed_dim,
                                            norm_layer=apply_layer_norm)

    def init_pos_emb(self, num_patches, drop_rate):
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

    def init_cls_token(self, embed_dim):
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def init_transformer(self):
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            self.block_type(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i],
                norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(self.depth)])

        self.norm = self.norm_layer(self.num_features)

    def patch_embedding(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def layer_forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        x = self.patch_embedding(x)
        x = self.block_forward(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def new_deit_tiny_patch16_224(pretrained=False, **kwargs):
    """
    the same as vit_base_patch16_224
    """
    default_kwargs = {
        'embed_dim': 192,
        'num_heads': 3,
    }
    default_kwargs.update(_cfg)
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['deit_base_patch16_224'], map_location='cpu')
        state_dict = modify_state_dict(vit, state_dict)
        vit.load_state_dict(state_dict, strict=False)
    return vit


def new_deit_small_patch16_224(pretrained=False, **kwargs):
    """
    the same as vit_base_patch16_224
    """
    default_kwargs = {
        'embed_dim': 384,
        'num_heads': 6,
    }
    default_kwargs.update(_cfg)
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['deit_small_patch16_224'], map_location='cpu')
        state_dict = modify_state_dict(vit, state_dict)
        vit.load_state_dict(state_dict, strict=False)
    return vit


def new_deit_base_patch16_224(pretrained=False, **kwargs):
    """
    the same as vit_base_patch16_224
    """
    default_kwargs = {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': 1000,
        'embed_dim': 786,
        'depth': [12],
        'mlp_ratio': 4.,
        'num_heads': 12,
        'qkv_bias': True,
        'representation_size': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.1,
        'drop_path_rate': 0.,
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['deit_base_patch16_224'], map_location='cpu')
        state_dict = modify_state_dict(vit, state_dict)
        vit.load_state_dict(state_dict, strict=False)
    return vit
