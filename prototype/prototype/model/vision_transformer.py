import math
import torch
import torch.nn as nn
from einops import rearrange  # noqa
from collections import OrderedDict
from .vit_hybrid_resnet import hybird_resnet50_3stage
from .layer import DropPath
from prototype.prototype.utils.trunc_normal_initializer import trunc_normal_


__all__ = ['vit_b32_224', 'vit_b16_224', 'hybird_vit_res50_b16_224', 'deit_tiny_b16_224',
           'deit_small_b16_224', 'deit_base_b16_224']


class GELU(nn.Module):
    """
    Gaussian Error Linear Units, based on
    `"Gaussian Error Linear Units (GELUs)" <https://arxiv.org/abs/1606.08415>`
    """
    def __init__(self, approximate=True):
        super(GELU, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate:
            cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) *
                                          (x + 0.044715 * torch.pow(x, 3))))
            return x * cdf
        else:
            return x * (torch.erf(x / math.sqrt(2)) + 1) / 2


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, activation=GELU):
        super(FeedForward, self).__init__()

        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.mlp2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dropout=0.1,
                 attention_dropout=0.1,
                 qkv_bias=True,
                 forward_type='einusm'):
        super(MultiHeadAttention, self).__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5  # 1/ sqrt(d_k)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.forward_type = forward_type

    def _forward1(self, x):
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # qk^T/ sqrt(d_k)

        attn = dots.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        return out

    def _forward2(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.to_out(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        if self.forward_type == 'einsum':
            return self._forward1(x)
        else:
            return self._forward2(x)


class Encoder1DBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, heads,
                 dropout, attention_dropout, drop_path, qkv_bias, activation, forward_type):
        super(Encoder1DBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout, attention_dropout, qkv_bias, forward_type)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = FeedForward(hidden_dim, mlp_dim, dropout, activation=activation)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

    def forward(self, x):

        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + residual

        y = self.norm2(x)
        y = self.feedforward(y)
        if self.drop_path is not None:
            y = self.drop_path(y)

        return x + y


class Encoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 depth,
                 mlp_dim,
                 heads,
                 dropout=0.1,
                 attention_dropout=0.1,
                 drop_path=0.1,
                 qkv_bias=True,
                 activation=GELU,
                 forward_type='einsum'):
        super(Encoder, self).__init__()

        encoder_layer = OrderedDict()
        for d in range(depth):
            encoder_layer['encoder_{}'.format(d)] = \
                Encoder1DBlock(hidden_dim, mlp_dim, heads, dropout, attention_dropout,
                               drop_path, qkv_bias, activation, forward_type)
        self.encoders = nn.Sequential(encoder_layer)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.encoders(x)
        x = self.encoder_norm(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 hidden_dim,
                 depth,
                 mlp_dim,
                 in_channel=3,
                 heads=8,
                 dropout=0.1,
                 attention_dropout=0.1,
                 classifier='token',
                 representation_size=None,
                 activation='gelu',
                 drop_path=0.1,
                 qkv_bias=True,
                 forward_type='einsum'):
        r"""
        Arguments:

        - in_channel (:obj:`int`): input channels
        - image_size (:obj:`int`): image size
        - patch_size (:obj:`int`): size of a patch
        - num_classes (:obj:`int`): number of classes
        - hidden_dim (:obj:`int`): embedding dimension of tokens
        - depth (:obj:`int`): number of encoder blocks
        - mlp_dim (:obj:`int`): hidden dimension in feedforward layer
        - heads (:obj:`int`): number of heads in multihead-attention
        - dropout (:obj:`float`): dropout rate after linear
        - attention_dropout (:obj:`float`): dropout rate after softmax in Acaled Dot-Product Attention
        - classifier (:obj:`str`): classifier type
        - representation_size
        - activation (:obj:`Moudle`): ReLU or GELU
        - drop_path (:obj:`float`): droppath rate after attention and feedforward
        - qkv_bias (:obj:`bool`): whether assign biases for qkv linear
        """

        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert classifier in ['token', 'gap']
        num_patches = (image_size // patch_size) ** 2

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.classifier = classifier
        self.representation_size = representation_size

        self.embedding = nn.Conv2d(in_channel, hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0)

        if classifier == 'token':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            trunc_normal_(self.cls_token, std=0.02)
        elif classifier == 'gap':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        trunc_normal_(self.pos_embedding, std=0.02)

        if activation == 'gelu':
            activation = GELU
        elif activation == 'relu':
            activation = nn.ReLU

        self.transformer = Encoder(
            hidden_dim, depth, mlp_dim, heads, dropout, attention_dropout,
            drop_path, qkv_bias, activation, forward_type)

        self.dropout = nn.Dropout(dropout)

        # if representation_size is not None
        # a pre-logits layer is inserted before classification head
        # it means transformer will be trained from scratch
        # otherwise the model is finetuned on new task
        if representation_size is not None:
            self.pre_logits = nn.Linear(hidden_dim, representation_size)
            self.tanh = nn.Tanh()
            self.head = nn.Linear(representation_size, num_classes)
        else:
            self.head = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_param_no_wd(self, fc=False, norm=False):

        no_wd = [self.pos_embedding, self.cls_token]
        no_wd.extend([self.embedding.weight, self.embedding.bias])
        for m in self.modules():
            if fc and isinstance(m, nn.Linear) and m.bias is not None:
                no_wd.append(m.bias)
            elif norm and isinstance(m, nn.LayerNorm):
                no_wd.append(m.bias)
                no_wd.append(m.weight)
        return no_wd

    def forward(self, img):
        x = self.embedding(img).reshape(img.shape[0], -1, self.num_patches).transpose(1, 2)
        # x shape: [B, N, K]

        if self.classifier == 'token':
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            # x shape: [B, N+1, K]

        x += self.pos_embedding
        x = self.dropout(x)

        # transformer main body
        x = self.transformer(x)

        # type of classifier
        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = torch.mean(x, dim=1, keepdim=False)

        if self.representation_size is None:
            # finetune on new task
            return self.head(x)
        else:
            # train from scratch
            return self.head(self.tanh(self.pre_logits(x)))


class HybirdVisitionTransformer(VisionTransformer):
    def __init__(self,
                 backbone,
                 image_size,
                 patch_size,
                 num_classes,
                 hidden_dim,
                 depth,
                 mlp_dim,
                 in_channel=3,
                 heads=8,
                 dropout=0.1,
                 attention_dropout=0.1,
                 classifier='token',
                 representation_size=None,
                 activation='gelu',
                 drop_path=0.1,
                 qkv_bias=True,
                 forward_type='einsum'):

        fake_input = torch.rand(1, in_channel, image_size, image_size)
        _, out_ch, out_size, _ = backbone(fake_input).size()
        super(HybirdVisitionTransformer, self).__init__(
                 out_size, patch_size, num_classes, hidden_dim,
                 depth, mlp_dim, out_ch, heads, dropout,
                 attention_dropout, classifier, representation_size,
                 activation, drop_path, qkv_bias, forward_type)
        self.backbone = backbone

    def forward(self, x):
        feature = self.backbone(x)
        return super(HybirdVisitionTransformer, self).forward(feature)


def vit_b32_224(**kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 32,
        'num_classes': 1000,
        'hidden_dim': 768,
        'depth': 12,
        'mlp_dim': 3072,
        'heads': 12,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token',
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def vit_b16_224(**kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 1000,
        'hidden_dim': 768,
        'depth': 12,
        'mlp_dim': 3072,
        'heads': 12,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token',
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def hybird_vit_res50_b16_224(**kwargs):
    backbone = hybird_resnet50_3stage()
    default_kwargs = {
        'image_size': 224,
        'patch_size': 1,
        'num_classes': 1000,
        'hidden_dim': 768,
        'depth': 12,
        'mlp_dim': 3072,
        'heads': 12,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token',
    }
    default_kwargs.update(kwargs)
    vit = HybirdVisitionTransformer(backbone, **default_kwargs)
    return vit


def deit_tiny_b16_224(**kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 1000,
        'hidden_dim': 192,
        'depth': 12,
        'mlp_dim': 768,
        'heads': 3,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token',
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def deit_small_b16_224(**kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 1000,
        'hidden_dim': 384,
        'depth': 12,
        'mlp_dim': 1536,
        'heads': 6,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token',
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def deit_base_b16_224(**kwargs):
    """
    the same as vit_b16_224
    """
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 1000,
        'hidden_dim': 768,
        'depth': 12,
        'mlp_dim': 3072,
        'heads': 12,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token',
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit
