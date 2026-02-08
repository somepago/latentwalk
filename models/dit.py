"""
Small DiT (Diffusion Transformer) for 64x64 image generation with DINO conditioning.
Uses adaLN-Zero for timestep conditioning and cross-attention for DINO feature conditioning.
"""

import torch
import torch.nn as nn
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero modulation, self-attention, cross-attention to DINO features, and FFN.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Cross-attention to DINO features
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # adaLN-Zero: 9 modulation params (shift, scale, gate for self-attn, cross-attn, ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )
        # Initialize gate params to zero so blocks start as identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, dino_features):
        """
        Args:
            x: [B, N, D] patch tokens
            c: [B, D] timestep conditioning
            dino_features: [B, T, D] DINO features for cross-attention
        """
        mod = self.adaLN_modulation(c).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa = mod[0], mod[1], mod[2]
        shift_ca, scale_ca, gate_ca = mod[3], mod[4], mod[5]
        shift_ff, scale_ff, gate_ff = mod[6], mod[7], mod[8]

        # Self-attention with adaLN-Zero
        h = modulate(self.norm1(x), shift_sa, scale_sa)
        h, _ = self.attn(h, h, h)
        x = x + gate_sa.unsqueeze(1) * h

        # Cross-attention to DINO features
        h = modulate(self.norm_cross(x), shift_ca, scale_ca)
        h, _ = self.cross_attn(h, dino_features, dino_features)
        x = x + gate_ca.unsqueeze(1) * h

        # FFN
        h = modulate(self.norm2(x), shift_ff, scale_ff)
        h = self.ffn(h)
        x = x + gate_ff.unsqueeze(1) * h

        return x


class DiT(nn.Module):
    """
    Small Diffusion Transformer for 64x64 image generation.

    Patchifies input images, processes with transformer blocks conditioned on
    timestep (adaLN-Zero) and DINO features (cross-attention), then unpatchifies.
    """

    def __init__(
        self,
        image_size=64,
        patch_size=16,
        in_channels=3,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Final layer
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.final_proj = nn.Linear(hidden_size, patch_dim)

        # Initialize
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

    def patchify(self, x):
        """[B, C, H, W] -> [B, N, patch_dim]"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H//p, W//p, C, p, p]
        x = x.reshape(B, self.num_patches, -1)  # [B, N, C*p*p]
        return x

    def unpatchify(self, x):
        """[B, N, patch_dim] -> [B, C, H, W]"""
        B = x.shape[0]
        p = self.patch_size
        h = w = self.image_size // p
        C = self.in_channels
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, h, p, w, p]
        x = x.reshape(B, C, self.image_size, self.image_size)
        return x

    def forward(self, x, t, dino_features):
        """
        Args:
            x: [B, C, H, W] noisy images
            t: [B] timesteps in [0, 1]
            dino_features: [B, T, 384] DINO features
        Returns:
            [B, C, H, W] predicted velocity
        """
        # Patchify and embed
        x = self.patch_embed(self.patchify(x)) + self.pos_embed

        # Timestep conditioning
        c = self.t_embedder(t)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, dino_features)

        # Final layer
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.final_proj(x)

        return self.unpatchify(x)


if __name__ == "__main__":
    model = DiT()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    dino = torch.randn(2, 16, 384)  # 56x56 DINO input -> 4x4 = 16 tokens
    out = model(x, t, dino)
    print(f"Input: {x.shape} -> Output: {out.shape}")
