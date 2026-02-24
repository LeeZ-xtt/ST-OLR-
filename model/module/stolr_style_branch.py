"""
ST-OLR 风格分支模块

包含统计Token化器、Token混合器和完整的风格分支实现。
从多尺度特征图中提取统计token，通过Transformer融合，最终生成风格嵌入和域分类logits。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List


class StatsTokenizer(nn.Module):
    """
    统计Token化器

    从多尺度特征图（f1, f2, f3）提取统计token。
    对每个尺度提取10个统计量：
    - 1个全局均值 + 1个全局方差（log-variance）
    - 4个区域均值 + 4个区域方差（log-variance，2×2区域划分）

    总共产生 3个尺度 × 10个统计量 = 30个token

    Args:
        token_dim: token的输出维度，默认128
    """

    def __init__(self, token_dim: int = 128):
        super().__init__()

        self.token_dim = token_dim

        # 为每个尺度创建投影层（Linear + LayerNorm）
        # f1: 64维, f2: 160维, f3: 320维
        # LayerNorm 确保不同尺度的 token 进入 Transformer 时激活范围一致
        self.proj_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(64, token_dim), nn.LayerNorm(token_dim)),
                nn.Sequential(nn.Linear(160, token_dim), nn.LayerNorm(token_dim)),
                nn.Sequential(nn.Linear(320, token_dim), nn.LayerNorm(token_dim)),
            ]
        )

        # 可学习的尺度编码 [3个尺度, 1, token_dim]，仅编码尺度身份
        # 通过 broadcast 加到该尺度所有10个token上
        self.scale_encodings = nn.Parameter(torch.empty(3, 1, token_dim))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 投影层 Linear 使用 Kaiming 初始化
        # fan_in 模式适合后续 LayerNorm，nonlinearity='linear' 因为后面没有激活函数
        for proj in self.proj_layers:
            nn.init.kaiming_normal_(
                proj[0].weight, mode="fan_in", nonlinearity="linear"
            )
            if proj[0].bias is not None:
                nn.init.zeros_(proj[0].bias)

        # scale_encodings 是加性偏置，小初始值合理
        nn.init.trunc_normal_(self.scale_encodings, std=0.02)

    def _extract_stats(
        self, f: torch.Tensor, use_region_stats: bool = True
    ) -> torch.Tensor:
        """
        从单个特征图提取统计token

        Args:
            f: 特征图 [B, C, H, W]
            use_region_stats: 是否使用区域统计（2×2划分），对于小尺寸特征图建议关闭

        Returns:
            stats: 统计量 [B, 10, C]（如果 use_region_stats=True）
                  或 [B, 2, C]（如果 use_region_stats=False）
        """
        B, C, H, W = f.shape

        # 1. 全局统计
        global_mean = f.mean(dim=[2, 3])  # [B, C]
        global_var = torch.log(f.var(dim=[2, 3]) + 1e-6)  # [B, C], log-variance

        if not use_region_stats:
            # 对于小尺寸特征图（如 f3），只使用全局统计
            all_stats = torch.stack([global_mean, global_var], dim=1)  # [B, 2, C]
            return all_stats

        # 2. 区域统计（2×2划分）
        H_half = H // 2
        W_half = W // 2

        # 使用自适应池化确保区域划分正确（处理奇数尺寸）
        regions = [
            f[:, :, :H_half, :W_half],  # 左上
            f[:, :, :H_half, W_half:],  # 右上
            f[:, :, H_half:, :W_half],  # 左下
            f[:, :, H_half:, W_half:],  # 右下
        ]

        region_stats = []
        for region in regions:
            region_mean = region.mean(dim=[2, 3])  # [B, C]
            region_var = torch.log(
                region.var(dim=[2, 3]) + 1e-6
            )  # [B, C], log-variance
            region_stats.extend([region_mean, region_var])

        # 3. 拼接所有统计量 [B, 10, C]
        all_stats = torch.stack(
            [global_mean, global_var, *region_stats], dim=1
        )  # [B, 10, C]

        return all_stats

    def forward(self, pyramid_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        提取统计token

        Args:
            pyramid_features: 包含 "f1", "f2", "f3" 的字典
                - f1: [B, 64, H/2, W/2]
                - f2: [B, 160, H/4, W/4]
                - f3: [B, 320, H/8, W/8] (典型尺寸约 10×10，只使用全局统计)

        Returns:
            tokens: 统计token [B, 22, token_dim]
                - f1: 10 tokens (全局 + 4区域)
                - f2: 10 tokens (全局 + 4区域)
                - f3: 2 tokens (仅全局)
        """
        tokens_list = []

        # f1 和 f2 使用区域统计，f3 只使用全局统计（避免小尺寸区域方差估计噪声大）
        use_region_flags = [True, True, False]

        for scale_idx, key in enumerate(["f1", "f2", "f3"]):
            f = pyramid_features[key]  # [B, C, H, W]

            # 提取统计量
            scale_stats = self._extract_stats(
                f, use_region_stats=use_region_flags[scale_idx]
            )

            # 线性投影到token_dim
            scale_tokens = self.proj_layers[scale_idx](
                scale_stats
            )  # [B, 10 or 2, token_dim]

            # 添加尺度编码（broadcast）
            if scale_tokens.shape[1] == 10:
                scale_tokens = (
                    scale_tokens + self.scale_encodings[scale_idx]
                )  # [B, 10, token_dim]
            else:
                # f3 只有 2 个 token，扩展尺度编码
                scale_tokens = (
                    scale_tokens + self.scale_encodings[scale_idx]
                )  # [B, 2, token_dim]

            tokens_list.append(scale_tokens)

        # 拼接所有尺度的token
        tokens = torch.cat(tokens_list, dim=1)  # [B, 22, token_dim]

        return tokens


class PatchTokenizer(nn.Module):
    """
    Patch Token 化器

    对每个尺度（f1/f2/f3）的特征图做 1×1 卷积投影 + 空间展平，
    输出原始 patch token 序列。可选地对空间尺寸过大的特征图做自适应下采样。

    设计决策：
    - 1×1 卷积不做任何空间混合，避免相邻像素语义结构泄漏
    - 使用 LayerNorm（而非 GroupNorm），因为 GroupNorm 会跨空间位置归一化
    - 不使用位置编码，保证 permutation-invariant

    Args:
        token_dim: token 输出维度，默认 128
        max_patches_per_scale: 单尺度最大 patch 数，默认 196
    """

    def __init__(self, token_dim: int = 128, max_patches_per_scale: int = 196):
        super().__init__()

        self.token_dim = token_dim
        self.max_patches = max_patches_per_scale

        # 三个尺度的 1×1 卷积投影（无偏置）
        # f1: 64 -> 128, f2: 160 -> 128, f3: 320 -> 128
        self.proj_layers = nn.ModuleList(
            [
                nn.Conv2d(64, token_dim, kernel_size=1, bias=False),
                nn.Conv2d(160, token_dim, kernel_size=1, bias=False),
                nn.Conv2d(320, token_dim, kernel_size=1, bias=False),
            ]
        )

        # 共享的 LayerNorm（在 forward 中对展平后的 tokens 应用）
        self.patch_ln = nn.LayerNorm(token_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # Conv2d 使用 kaiming_normal，fan_out 模式适合投影到更低维
        for proj in self.proj_layers:
            nn.init.kaiming_normal_(proj.weight, mode="fan_out", nonlinearity="linear")

        # LayerNorm 默认初始化 (gamma=1, beta=0)

    def forward(self, pyramid_features: Dict[str, torch.Tensor]) -> list:
        """
        提取 patch tokens

        Args:
            pyramid_features: 包含 "f1", "f2", "f3" 的字典
                - f1: [B, 64, H/2, W/2]
                - f2: [B, 160, H/4, W/4]
                - f3: [B, 320, H/8, W/8]

        Returns:
            patch_tokens_per_scale: List[Tensor]，每个元素 [B, n_patches_i, token_dim]
        """
        result = []
        keys = ["f1", "f2", "f3"]

        for scale_idx, key in enumerate(keys):
            f = pyramid_features[key]  # [B, C, H, W]

            B, C, H, W = f.shape
            n_patches = H * W

            # 自适应下采样：如果 patch 数超过 max_patches
            if n_patches > self.max_patches:
                target_size = int(math.sqrt(self.max_patches))
                f = F.adaptive_avg_pool2d(f, (target_size, target_size))
                B, C, H, W = f.shape

            # 1×1 卷积投影
            f_proj = self.proj_layers[scale_idx](f)  # [B, token_dim, H', W']

            # 空间展平 + permute: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = f_proj.shape
            tokens = f_proj.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # LayerNorm
            tokens = self.patch_ln(tokens)  # [B, HW, token_dim]

            result.append(tokens)

        return result


class StyleCrossAttnAggregator(nn.Module):
    """
    风格 Cross-Attention 聚合器

    用可学习的 style queries 对每个尺度的原始 patch tokens 做 cross-attention，
    将数百个 patch 压缩为 K_p 个风格 token。

    设计决策：
    - Queries 按尺度独立（不同尺度捕捉不同风格因子）
    - Attention 权重跨尺度共享（减少参数量，正则化）
    - 使用 PyTorch nn.MultiheadAttention（数值稳定，支持 flash attention）
    - 输出加 LayerNorm 统一尺度

    Args:
        token_dim: token 维度，默认 128
        n_queries: 每个尺度的 query 数，默认 8
        n_heads: 注意力头数，默认 4
        n_scales: 尺度数，默认 3
        dropout: Dropout 率，默认 0.1
    """

    def __init__(
        self,
        token_dim: int = 128,
        n_queries: int = 8,
        n_heads: int = 4,
        n_scales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.n_scales = n_scales

        # 可学习的 style queries，按尺度独立
        self.style_queries = nn.Parameter(torch.empty(n_scales, n_queries, token_dim))

        # Cross-attention（跨尺度共享权重）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 输出归一化
        self.out_norm = nn.LayerNorm(token_dim)

        # 尺度编码（加到聚合后的 patch tokens 上）
        self.scale_encodings = nn.Parameter(torch.empty(n_scales, 1, token_dim))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # style_queries: 小初始值使初始 attention 接近均匀分布
        nn.init.trunc_normal_(self.style_queries, std=0.02)

        # scale_encodings: 与 StatsTokenizer 一致
        nn.init.trunc_normal_(self.scale_encodings, std=0.02)

        # MultiheadAttention 内部使用 PyTorch 默认 Xavier uniform

    def forward(self, patch_tokens_per_scale: list) -> torch.Tensor:
        """
        Cross-attention 聚合

        Args:
            patch_tokens_per_scale: List[Tensor]，每个元素 [B, n_patches_i, token_dim]

        Returns:
            aggregated_tokens: [B, n_scales * n_queries, token_dim] = [B, 24, 128]
        """
        result = []

        # 获取 batch size
        B = patch_tokens_per_scale[0].shape[0]

        for i in range(self.n_scales):
            patches_i = patch_tokens_per_scale[i]  # [B, n_patches_i, token_dim]

            # 扩展 queries 到 batch 维度
            queries_i = self.style_queries[i]  # [n_queries, token_dim]
            queries_i = queries_i.unsqueeze(0).expand(
                B, -1, -1
            )  # [B, n_queries, token_dim]

            # Cross-attention: queries attend to patches
            attn_out, _ = self.cross_attn(
                query=queries_i,  # [B, n_queries, token_dim]
                key=patches_i,  # [B, n_patches_i, token_dim]
                value=patches_i,  # [B, n_patches_i, token_dim]
            )  # attn_out: [B, n_queries, token_dim]

            # 残差连接：保留 queries 信息，梯度可直达 style_queries
            attn_out = queries_i + attn_out

            # 加尺度编码（在LayerNorm之前，避免破坏归一化分布）
            attn_out = attn_out + self.scale_encodings[i]  # broadcast

            # 归一化
            attn_out = self.out_norm(attn_out)

            result.append(attn_out)

        aggregated_tokens = torch.cat(result, dim=1)  # [B, 3*n_queries, token_dim]
        return aggregated_tokens


class TokenMixer(nn.Module):
    """
    Token混合器

    使用Transformer Encoder融合统计token，建模不同尺度和区域之间的依赖关系。

    Args:
        token_dim: token维度，默认128
        n_layers: Transformer层数，默认2
        n_heads: 注意力头数，默认4
    """

    def __init__(self, token_dim: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()

        self.token_dim = token_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=n_heads,
                    dim_feedforward=token_dim * 4,
                    activation="gelu",
                    norm_first=True,
                    batch_first=True,
                    dropout=0.1,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        融合统计token

        Args:
            tokens: 输入token [B, 30, token_dim]

        Returns:
            mixed_tokens: 融合后的token [B, 30, token_dim]
        """
        x = tokens  # [B, 30, token_dim]

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return x


class AttentionPooling(nn.Module):
    """
    注意力池化层

    使用可学习的 query 对 token 序列做注意力加权池化，替代简单的 mean pooling。
    解决 mean pooling 无法区分 token 重要性的问题。

    Args:
        token_dim: token 维度，默认 128
    """

    def __init__(self, token_dim: int = 128):
        super().__init__()
        self.token_dim = token_dim
        self.scale = token_dim**-0.5

        # 可学习的聚合 query [1, 1, token_dim]
        self.query = nn.Parameter(torch.empty(1, 1, token_dim))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.query)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        注意力池化

        Args:
            tokens: 输入 tokens [B, n_tokens, token_dim]

        Returns:
            pooled: 池化后的特征 [B, token_dim]
        """
        B = tokens.shape[0]

        # 扩展 query 到 batch 维度
        query = self.query.expand(B, -1, -1)  # [B, 1, token_dim]

        # 计算注意力分数
        scores = torch.matmul(query, tokens.transpose(-2, -1)) * self.scale
        # scores: [B, 1, n_tokens]

        # Softmax 获得注意力权重
        weights = F.softmax(scores, dim=-1)  # [B, 1, n_tokens]

        # 加权求和
        pooled = torch.matmul(weights, tokens).squeeze(1)  # [B, token_dim]

        return pooled
