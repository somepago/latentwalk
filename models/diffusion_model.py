"""
MVP implementation of Sana 600M model for training.
Standalone version with all necessary components copied from parent repository.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from typing import Union, Tuple, List, Optional
from itertools import repeat
from collections.abc import Iterable


# ============================================================================
# Utility functions from diffusion.model.utils
# ============================================================================

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def val2tuple(x, n: int) -> Tuple:
    """Convert a value to a tuple of length n."""
    if isinstance(x, (tuple, list)):
        assert len(x) == n
        return tuple(x)
    else:
        return (x,) * n


def get_same_padding(kernel_size: int) -> int:
    """Calculate padding to maintain spatial dimensions."""
    if kernel_size % 2 == 0:
        raise ValueError("Only odd kernel sizes are supported")
    return (kernel_size - 1) // 2


def auto_grad_checkpoint(module, *args, **kwargs):
    """Placeholder for gradient checkpointing - not used in inference."""
    return module(*args, **kwargs)


# ============================================================================
# Basic modules from diffusion.model.nets.sana_blocks
# ============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ============================================================================
# Norm layers from diffusion.model.norms
# ============================================================================

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

    Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        return (self.weight * self._norm(x.float())).type_as(x)


# ============================================================================
# Building blocks from diffusion.model.act and diffusion.model.norms
# ============================================================================

def build_act(act: Union[str, None], inplace: bool = True):
    """Build activation layer."""
    if act is None:
        return nn.Identity()
    elif act == "relu":
        return nn.ReLU(inplace=inplace)
    elif act == "silu":
        return nn.SiLU(inplace=inplace)
    elif act == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {act}")


def build_norm(norm: Union[str, None], num_features: int, **kwargs):
    """Build normalization layer."""
    if norm is None:
        return nn.Identity()
    elif norm == "bn2d":
        return nn.BatchNorm2d(num_features, **kwargs)
    elif norm == "ln2d":
        return nn.LayerNorm(num_features, **kwargs)
    else:
        raise ValueError(f"Unknown normalization: {norm}")


# ============================================================================
# Conv layers from diffusion.model.nets.basic_modules
# ============================================================================

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: int or None = None,
        use_bias=False,
        dropout=0.0,
        norm="bn2d",
        act="relu",
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features=None,
        kernel_size=3,
        stride=1,
        padding: int or None = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_features = out_features or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = build_act(act[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_features,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        C_out = self.point_conv.out_dim
        x = x.reshape(B, C_out, N).permute(0, 2, 1)

        return x


# ============================================================================
# Attention modules from diffusion.model.nets.sana_blocks
# ============================================================================

class LiteLA(nn.Module):
    """Lightweight linear attention from diffusion.model.nets.sana_blocks"""

    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        qk_norm=False,
        norm_eps=1e-5,
    ):
        super().__init__()
        heads = heads or int(out_dim // dim * heads_ratio)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps

        # QKV projection
        self.qkv = nn.Linear(in_dim, out_dim * 3, bias=use_bias)

        # Output projection
        self.proj = nn.Linear(out_dim, out_dim)

        # Activation for linear attention
        self.kernel_func = nn.ReLU(inplace=False)

        # QK normalization
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # For compatibility
        self.num_heads = heads
        self.head_dim = self.dim

    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(self, x: torch.Tensor, mask=None, HW=None, image_rotary_emb=None, block_id=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)
        k = k.reshape(B, C // self.dim, self.dim, N)
        v = v.reshape(B, C // self.dim, self.dim, N)

        out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)
        out = self.proj(out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(B, -1, 2, C)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(B, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(q.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)

        x = x.contiguous().view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# ============================================================================
# Embedding layers from diffusion.model.nets.sana_blocks
# ============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None, mask=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)

        caption = self.y_proj(caption)

        return caption


class PatchEmbedMS(nn.Module):
    """2D Image to Patch Embedding for multi-scale"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        kernel_size=None,
        padding=0,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        kernel_size = kernel_size or patch_size
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # Debug: Check input to patch embedding
        if torch.isnan(x).any():
            print(f"NaN detected in patch embedding input")
            print(f"Input stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        
        # Debug: Check convolution weights
        if torch.isnan(self.proj.weight).any():
            print(f"NaN detected in patch embedding convolution weights")
            print(f"Weight stats: min={self.proj.weight.min().item()}, max={self.proj.weight.max().item()}, mean={self.proj.weight.mean().item()}")
        
        x = self.proj(x)
        
        # Debug: Check after convolution
        if torch.isnan(x).any():
            print(f"NaN detected after patch embedding convolution")
            print(f"Conv output stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            
            # Debug: Check after flatten
            if torch.isnan(x).any():
                print(f"NaN detected after patch embedding flatten")
                print(f"Flatten output stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        
        x = self.norm(x)
        
        # Debug: Check after normalization
        if torch.isnan(x).any():
            print(f"NaN detected after patch embedding normalization")
            print(f"Norm output stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================================
# Positional embedding functions from diffusion.model.nets.sana
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / pe_interpolation
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# ============================================================================
# Main model blocks
# ============================================================================

class SanaBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        qk_norm=False,
        cross_norm=False,
        attn_type="linear",
        ffn_type="glumbconv",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_attn_type="flash",
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if attn_type == "linear":
            # linear self attention
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        # Use MultiHeadCrossAttention
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Use GLUMBConv as the default FFN for Sana
        if ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu,
                drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, mask=None, HW=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        x = x + self.drop_path(
            gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW)
        )
        # Handle mask properly - it might be a list of lengths
        if isinstance(mask, list):
            # mask is y_lens, don't pass it to cross_attn
            x = x + self.cross_attn(x, y, None)
        else:
            x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), HW=HW))

        return x


class Sana_600M(nn.Module):
    """
    Sana 600M model for text-to-image generation using flow matching.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=1,
        in_channels=32,  # Sana uses DC-AE VAE with 32 channels!
        hidden_size=1152,  # 600M model uses 1152
        depth=28,  # 600M model uses 28 layers
        num_heads=16,  # 600M model uses 16 heads
        mlp_ratio=2.5,
        caption_channels=2304,
        model_max_length=300,
        qk_norm=False,  # 600M uses False by default
        cross_norm=False,
        y_norm=True,
        y_norm_scale_factor=0.01,
        use_pe=False,  # Sana 600M uses False by default!
        pe_interpolation=1.0,
        pred_sigma=False,
        learn_sigma=False,
        class_dropout_prob=0.1,
        attn_type="linear",
        ffn_type="glumbconv",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_attn_type="flash",
        pos_embed_type="sincos",
        timestep_norm_scale_factor=1.0,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.use_pe = use_pe
        self.pos_embed_type = pos_embed_type
        self.y_norm = y_norm
        self.y_norm_scale_factor = y_norm_scale_factor
        self.timestep_norm_scale_factor = timestep_norm_scale_factor

        # Use PatchEmbedMS
        kernel_size = patch_size
        self.x_embedder = PatchEmbedMS(patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True)

        # Use TimestepEmbedder
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Calculate base size for positional embeddings
        self.base_size = input_size // self.patch_size

        # Use CaptionEmbedder
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        # Use RMSNorm for y_norm
        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=1e-5)

        # T block for timestep embedding
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Transformer blocks
        drop_path = [x.item() for x in torch.linspace(0, 0.0, depth)]  # no drop path for MVP
        self.blocks = nn.ModuleList(
            [
            SanaBlock(
                    hidden_size,
                    num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                qk_norm=qk_norm,
                cross_norm=cross_norm,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_attn_type=cross_attn_type,
            )
            for i in range(depth)
            ]
        )

        # Use T2IFinalLayer
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        # Will be set dynamically
        self.h = self.w = 0
        self.pos_embed_ms = None

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if hasattr(self.x_embedder.proj, 'bias') and self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = self.h
        w = self.w
        # Debug info removed for clean output
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, timestep, y, mask=None, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations)
        timestep: (N,) tensor of diffusion timesteps
        y: (N, L, D) or (N, 1, L, D) tensor of caption embeddings
        mask: (N, L) attention mask for captions
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # Debug: Check input values
        if torch.isnan(x).any():
            print(f"NaN detected in input x")
        if torch.isnan(timestep).any():
            print(f"NaN detected in timestep")
        if torch.isnan(y).any():
            print(f"NaN detected in input y")

        # Store HW for unpatchify
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        # Debug: Check input before patch embedding
        if torch.isnan(x).any():
            print(f"NaN detected in input before patch embedding")
            print(f"Input stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        
        # Patch embedding
        x = self.x_embedder(x)  # (N, T, D)
        
        # Debug: Check after patch embedding
        if torch.isnan(x).any():
            print(f"NaN detected after patch embedding")
            print(f"Patch embedding output stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
            # Return a tensor with the correct output shape instead of the intermediate shape
            # Use the original input dimensions for the output shape
            return torch.full((x.shape[0], self.out_channels, self.h * self.patch_size, self.w * self.patch_size), float('nan'), device=x.device, dtype=x.dtype)

        # Add positional embedding
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x.shape[1:]:
                    self.pos_embed_ms = (
                        torch.from_numpy(
                get_2d_sincos_pos_embed(
                    self.hidden_size,
                                (self.h, self.w),
                                pe_interpolation=self.pe_interpolation,
                                base_size=self.base_size,
                            )
                        )
                        .float()
                        .unsqueeze(0)
                        .to(x.device)
                        .to(x.dtype)
                    )
                x = x + self.pos_embed_ms

        # Time embedding
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        
        # Debug: Check time embedding
        if torch.isnan(t).any():
            print(f"NaN detected in time embedding")
            # Return a tensor with the correct output shape instead of the intermediate shape
            return torch.full((x.shape[0], self.out_channels, self.h * self.patch_size, self.w * self.patch_size), float('nan'), device=x.device, dtype=x.dtype)
            
        t0 = self.t_block(t)  # (N, 6*D)
        
        # Debug: Check t0
        if torch.isnan(t0).any():
            print(f"NaN detected in t0")
            # Return a tensor with the correct output shape instead of the intermediate shape
            return torch.full((x.shape[0], self.out_channels, self.h * self.patch_size, self.w * self.patch_size), float('nan'), device=x.device, dtype=x.dtype)

        # Caption embedding
        # Handle different input shapes for y
        if y.dim() == 3:
            # y is (N, L, D)
            if y.shape[1] == 1:
                # y is (N, 1, D) - expand to (N, 1, L, D) where L=1
                y = y.unsqueeze(2)
            else:
                # y is (N, L, D) - reshape to (N, 1, L, D)
                y = y.unsqueeze(1)

        y = self.y_embedder(y, self.training, mask=mask)  # (N, 1, L, D)

        # Apply y normalization
        if self.y_norm:
            y = self.attention_y_norm(y)

        # Process mask and prepare y for cross-attention
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, y, t0, y_lens, HW=(self.h, self.w))

        # Final layer
        x = self.final_layer(x, t)  # (N, T, patch_size**2 * out_channels)
        # Debug info removed for clean output
        
        # Unpatchify
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        # Debug info removed for clean output

        # Split learned variance if applicable
        if self.pred_sigma:
            x, _ = torch.split(x, self.in_channels, dim=1)

        # Debug info removed for clean output
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        DPM solver doesn't need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out

    @property
    def dtype(self):
        """Get model's current dtype from its parameters."""
        return next(self.parameters()).dtype


if __name__ == "__main__":
    # Test the model
    model = Sana_600M()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 32, 32, 32)  # 32 channels for DC-AE VAE
    timestep = torch.randint(0, 1000, (batch_size,))
    y = torch.randn(batch_size, 300, 2304)
    mask = torch.ones(batch_size, 300)

    with torch.no_grad():
        output = model(x, timestep, y, mask)
        # Debug info removed for clean output