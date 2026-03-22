"""MegaLoc: One Retrieval to Place Them All

Self-contained MegaLoc model for Netryx integration.
Paper: https://arxiv.org/abs/2502.17237
License: MIT
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tfm


def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    M = M / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class FeatureAggregator(nn.Module):
    def __init__(self, num_channels=1536, num_clusters=64, cluster_dim=128,
                 token_dim=256, mlp_dim=512, dropout=0.3):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.mlp_dim = mlp_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, self.mlp_dim), nn.ReLU(),
            nn.Linear(self.mlp_dim, self.token_dim))
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1), dropout, nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.cluster_dim, 1))
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1), dropout, nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.num_clusters, 1))
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x, t = x
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        f = torch.cat([
            F.normalize(t, p=2, dim=-1),
            F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
        ], dim=-1)
        return F.normalize(f, p=2, dim=-1)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=518, patch_size=14, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_value=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = TransformerMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init_value=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2Backbone(nn.Module):
    def __init__(self, image_size=518, patch_size=14, in_channels=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_channels = embed_dim
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size,
                                          in_channels=in_channels, embed_dim=embed_dim)
        self.interpolate_offset = 0.1
        self.interpolate_antialias = False
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy), mode="bicubic", antialias=self.interpolate_antialias)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, images):
        B, _, H, W = images.shape
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        patch_features = patch_tokens.contiguous().reshape(
            B, H // self.patch_size, W // self.patch_size, self.embed_dim
        ).permute(0, 3, 1, 2).contiguous()
        return patch_features, cls_token


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


class Aggregator(nn.Module):
    def __init__(self, feat_dim, agg_config, salad_out_dim):
        super().__init__()
        self.agg = FeatureAggregator(**agg_config)
        self.linear = nn.Linear(salad_out_dim, feat_dim)

    def forward(self, x):
        x = self.agg(x)
        return self.linear(x)


class MegaLoc(nn.Module):
    """MegaLoc model for visual place recognition."""

    def __init__(self, feat_dim=8448, num_clusters=64, cluster_dim=256,
                 token_dim=256, mlp_dim=512):
        super().__init__()
        self.backbone = DINOv2Backbone()
        self.salad_out_dim = num_clusters * cluster_dim + token_dim
        self.aggregator = Aggregator(
            feat_dim=feat_dim,
            agg_config={
                "num_channels": self.backbone.num_channels,
                "num_clusters": num_clusters,
                "cluster_dim": cluster_dim,
                "token_dim": token_dim,
                "mlp_dim": mlp_dim,
            },
            salad_out_dim=self.salad_out_dim,
        )
        self.feat_dim = feat_dim
        self.l2norm = L2Norm()

    def forward(self, images):
        b, c, h, w = images.shape
        if h % 14 != 0 or w % 14 != 0:
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            images = tfm.resize(images, [h, w], antialias=True)
        features = self.aggregator(self.backbone(images))
        features = self.l2norm(features)
        return features
