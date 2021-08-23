import torch
import torch.nn.functional as F
from .layers.helpers import replace_batchnorm

from .layers import Linear_BN, BN_Linear, AttentionSubsample, Attention, b16


__all__ = ['LeViT_128S', 'LeViT_128', 'LeViT_192', 'LeViT_256', 'LeViT_384']


class Hswish(torch.nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=Hswish,
                 mlp_activation=Hswish,
                 distillation=True,
                 drop_path=0,
                 fuse=False):
        super().__init__()
        global FLOPS_COUNTER

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.fuse = fuse

        self.patch_embed = hybrid_backbone

        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
        self.blocks = torch.nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


def LeViT_128S(**kwargs):
    embed_dim = [128, 256, 384]
    act = Hswish
    model = LeViT(
        **kwargs,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=[4, 6, 8],
        key_dim=[16, 16, 16],
        depth=[2, 3, 4],
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 16, embed_dim[0] // 16, 4, 2, 2],
            ['Subsample', 16, embed_dim[1] // 16, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        drop_path=0
    )
    if model.fuse:
        replace_batchnorm(model)
    return model


def LeViT_128(**kwargs):
    embed_dim = [128, 256, 384]
    act = Hswish
    model = LeViT(
        **kwargs,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=[4, 8, 12],
        key_dim=[16, 16, 16],
        depth=[4, 4, 4],
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 16, embed_dim[0] // 16, 4, 2, 2],
            ['Subsample', 16, embed_dim[1] // 16, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        drop_path=0
    )
    if model.fuse:
        replace_batchnorm(model)
    return model

def LeViT_192(**kwargs):
    embed_dim = [192, 288, 384]
    act = Hswish
    model = LeViT(
        **kwargs,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=[3, 5, 6],
        key_dim=[32, 32, 32],
        depth=[4, 4, 4],
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 32, embed_dim[0] // 32, 4, 2, 2],
            ['Subsample', 32, embed_dim[1] // 32, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        drop_path=0
    )
    if model.fuse:
        replace_batchnorm(model)
    return model


def LeViT_256(**kwargs):
    embed_dim = [256, 384, 512]
    act = Hswish
    model = LeViT(
        **kwargs,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=[4, 6, 8],
        key_dim=[32, 32, 32],
        depth=[4, 4, 4],
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 32, embed_dim[0] // 32, 4, 2, 2],
            ['Subsample', 32, embed_dim[1] // 32, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        drop_path=0
    )
    if model.fuse:
        replace_batchnorm(model)
    return model


def LeViT_384(**kwargs):
    embed_dim = [384, 512, 768]
    act = Hswish
    model = LeViT(
        **kwargs,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=[6, 9, 12],
        key_dim=[32, 32, 32],
        depth=[4, 4, 4],
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 32, embed_dim[0] // 32, 4, 2, 2],
            ['Subsample', 32, embed_dim[1] // 32, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        drop_path=0.1
    )
    if model.fuse:
        replace_batchnorm(model)
    return model
