"""BobbyExtractor — tile-embedding CNN + self-attention feature extractor.

Tile IDs → Embedding(64, 32) → reshape to (B, 32, 16, 16) → CNN+ResBlock
→ CNN+ResBlock → 4-head self-attention over the 16×16 token grid
→ adaptive pool → concat with an MLP over 9 normalised scalars
→ 256-dim features.

net_arch=[] (no extra MLP head — the extractor is the head).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return F.relu(x + self.net(x))


class _TileAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens.permute(0, 2, 1).view(B, C, H, W)


class BobbyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        EMBED_DIM = 32
        self.embed = nn.Embedding(64, EMBED_DIM)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(EMBED_DIM, 64, kernel_size=3, padding=1), nn.ReLU(),
            _ResBlock(64),
            nn.AdaptiveAvgPool2d(4),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            _ResBlock(128),
        )
        self.attn = _TileAttention(embed_dim=128, num_heads=4)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten())
        self.scalar_net = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.fuse = nn.Sequential(nn.Linear(512 + 64, features_dim), nn.ReLU())

    def forward(self, obs: dict) -> torch.Tensor:
        tiles = obs["tiles"].view(-1, 256).round().long().clamp(0, 63)
        B = tiles.size(0)
        e = self.embed(tiles)
        e = e.view(B, 16, 16, -1).permute(0, 3, 1, 2)
        x = self.cnn1(e)
        x = self.cnn2(x)
        x = self.attn(x)
        cnn_out = self.pool(x)

        def _proc_scalar(key, dim):
            val = obs[key]
            if val.dim() > 2:
                val = val.flatten(1)
            if val.shape[1] == dim:
                return val.argmax(1, keepdim=True).float() / float(dim - 1)
            return val.view(B, -1).float()

        px = _proc_scalar("player_x", 16)
        py = _proc_scalar("player_y", 16)
        cc = _proc_scalar("carrot_count", 64)
        ct = _proc_scalar("carrot_total", 64)
        ec = _proc_scalar("egg_count", 64)
        et = _proc_scalar("egg_total", 64)
        keys = obs["keys"].view(B, 3).float()

        # Completion signals: explicit ratio + binary flag so the network
        # can clearly distinguish "collecting" from "seek exit" phase.
        total = ct + et + 1e-8          # avoid div-by-zero
        collected = cc + ec
        completion_ratio = collected / total          # 0→1 as items collected
        all_collected = (completion_ratio >= 1.0 - 1e-6).float()  # binary flag

        scalars = torch.cat(
            [px, py, cc, ct, ec, et, keys, completion_ratio, all_collected],
            dim=1,
        )
        sc_out = self.scalar_net(scalars)
        return self.fuse(torch.cat([cnn_out, sc_out], dim=1))
