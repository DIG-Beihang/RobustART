import torch
import torch.nn as nn
from .vit_base import VisionTransformer
from .layers import VanillaMlp, DropPath


class MixerBlock(nn.Module):

    def __init__(self, num_patch, dim, token_mlp_ratio, channel_mlp_ratio,
                 drop=0., drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        token_mlp_dim = round(dim * token_mlp_ratio)
        channel_mlp_dim = round(dim * channel_mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mix = VanillaMlp(num_patch, token_mlp_dim, num_patch, act_layer, drop)
        self.channel_mix = VanillaMlp(dim, channel_mlp_dim, dim, act_layer, drop)

    def forward(self, x):

        y = self.norm1(x).transpose(1, 2)
        y = self.drop_path(self.token_mix(y)).transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        x = x + self.drop_path(self.channel_mix(y))
        return x


class MLPMixer(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 token_mlp_ratio=0.5, channel_mlp_ratio=4.0, representation_size=None, drop_rate=0., drop_path_rate=0.):
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
            weight_init: (str): weight init scheme
        """
        qkv_bias = False
        qk_scale = None
        num_heads = None
        attn_drop_rate = None
        mlp_ratio = token_mlp_ratio
        self.token_mlp_ratio = token_mlp_ratio
        self.channel_mlp_ratio = channel_mlp_ratio
        super(MLPMixer, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                       num_heads, mlp_ratio, qkv_bias, qk_scale, representation_size,
                                       drop_rate, attn_drop_rate, drop_path_rate)

        # Weight init
        # raise NotImplementedError

    def get_param_no_wd(self, fc=False, norm=False):
        no_wd = []
        return no_wd

    def init_block_type(self):
        self.block_type = MixerBlock

    def init_pos_emb(self, num_patches, drop_rate):
        pass

    def init_cls_token(self, embed_dim):
        pass

    def init_transformer(self):
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            self.block_type(
                num_patch=self.patch_embed.num_patches, dim=self.embed_dim,
                token_mlp_ratio=self.token_mlp_ratio, channel_mlp_ratio=self.channel_mlp_ratio,
                drop=self.drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(self.depth)])
        self.norm = self.norm_layer(self.embed_dim)

    def patch_embedding(self, x):
        x = self.patch_embed(x)
        return x

    def block_forward(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        x = self.patch_embedding(x)
        x = self.block_forward(x)
        return self.pre_logits(x.mean(dim=1))


def mixer_b16_224(**kwargs):
    return MLPMixer(img_size=224, patch_size=16, embed_dim=768, depth=12)


def mixer_L16_224(**kwargs):
    return MLPMixer(img_size=224, patch_size=16, embed_dim=1024, depth=24)
