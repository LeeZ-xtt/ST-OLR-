"""
Experiment B.1 Model - 联合 Stiefel 流形 + HSIC 独立性约束架构

该模型使用联合正交基底和 HSIC 独立性约束进行域自适应小样本学习。
核心改进：
1. 联合正交矩阵 Q ∈ St(320, 640) - 语义和风格子空间天然正交
2. HSIC 独立性损失 - 替代交叉相关损失，实现有效特征解耦
3. 潜码编码器 - 将高维特征编码为低秩潜码表示

使用方法:
    from model.exp_b1_model import ExpB1Model
    model = ExpB1Model()
    outputs = model(support_images, support_labels, query_images, n_way,
                    support_domains, query_domains)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from model.backbone.intrinsic_encoder_v2 import IntrinsicEncoder
from model.module.prototype import PrototypeNetwork
from model.module.stolr_style_branch import (
    StatsTokenizer,
    TokenMixer,
    PatchTokenizer,
    StyleCrossAttnAggregator,
    AttentionPooling,
)
from model.module.joint_stiefel_basis import JointStiefelBasis
from model.module.latent_encoder import LatentEncoder


class ExpB1Model(nn.Module):
    """
    实验 B.1 模型: ST-OLR 双分支架构

    Args:
        metric: 原型网络的距离度量 ('euclidean' 或 'cosine')
        intrinsic_encoder_drop_rate: 本征编码器ResNet12残差块Dropout率
        sem_olr_rank: 语义OLR投影器的秩，默认256
        sem_olr_mlp_hidden: 语义OLR投影器的MLP隐藏层维度，默认512
        style_olr_rank: 风格OLR投影器的秩，默认64
        style_olr_mlp_hidden: 风格OLR投影器的MLP隐藏层维度，默认256
        token_dim: Token维度，默认128
        n_domains: 域的数量，默认4
        detach_style_inputs: 是否detach风格分支的输入，默认True
    """

    def __init__(
        self,
        metric: str = "euclidean",
        intrinsic_encoder_drop_rate: float = 0.2,
        sem_olr_rank: int = 128,
        sem_olr_mlp_hidden: int = 512,
        style_olr_rank: int = 32,
        style_olr_mlp_hidden: int = 256,
        token_dim: int = 128,
        n_domains: int = 4,
        detach_style_inputs: bool = True,
        n_transformer_layers: int = 2,
        n_attention_heads: int = 4,
    ):
        """初始化ExpB1Model - ST-OLR双分支架构"""
        super().__init__()

        self.metric = metric
        self.detach_style_inputs = detach_style_inputs
        self.sem_olr_rank = sem_olr_rank
        self.style_olr_rank = style_olr_rank
        self.token_dim = token_dim
        self.n_domains = n_domains

        # 本征编码器（返回金字塔特征）
        self.backbone = IntrinsicEncoder(
            drop_rate=intrinsic_encoder_drop_rate, dilated=True
        )

        # === 新架构：联合 Stiefel 流形 + HSIC 独立性约束 ===
        # 联合正交基底（语义和风格子空间天然正交）
        self.joint_basis = JointStiefelBasis(
            ambient_dim=640, rank_c=sem_olr_rank, rank_s=style_olr_rank
        )

        self.sem_encoder = LatentEncoder(in_dim=640, rank=sem_olr_rank)

        self.sty_encoder = LatentEncoder(in_dim=token_dim, rank=style_olr_rank)

        self.stats_tokenizer = StatsTokenizer(token_dim=token_dim)
        self.patch_tokenizer = PatchTokenizer(
            token_dim=token_dim, max_patches_per_scale=196
        )
        self.cross_attn_agg = StyleCrossAttnAggregator(
            token_dim=token_dim,
            n_queries=8,
            n_heads=n_attention_heads,
            n_scales=3,
            dropout=0.1,
        )
        self.token_mixer = TokenMixer(
            token_dim=token_dim,
            n_layers=n_transformer_layers,
            n_heads=n_attention_heads,
        )
        self.attn_pool = AttentionPooling(token_dim=token_dim)

        # === 旧架构（保留以便回滚）===
        # 语义分支组件
        # self.sem_olr = OLRProjector(
        #     in_dim=640,
        #     rank=sem_olr_rank,
        #     mlp_hidden=sem_olr_mlp_hidden
        # )
        self.proto_head = PrototypeNetwork(metric=metric, temperature=10.0)

        # 风格分支
        # self.style_branch = STOLRStyleBranch(
        #     token_dim=token_dim,
        #     n_domains=n_domains,
        #     detach_inputs=detach_style_inputs
        # )

        # 域分类头（使用 u_s 进行域分类，64 维潜码更紧凑）
        self.domain_head = nn.Linear(style_olr_rank, n_domains)

        self.mode = "train"

        # 初始化域分类头
        nn.init.trunc_normal_(self.domain_head.weight, std=0.02)
        nn.init.zeros_(self.domain_head.bias)

        # 打印模型信息
        self._print_model_info()

    def _print_model_info(self):
        print("[INFO] Experiment B.1 Model (联合 Stiefel + HSIC) 配置:")
        print("   骨干网络: IntrinsicEncoder (金字塔特征)")
        print(
            f"   联合正交基底: JointStiefelBasis (640×{self.sem_olr_rank + self.style_olr_rank})"
        )
        print(f"   语义编码器: LatentEncoder (640 -> {self.sem_olr_rank})")
        print(
            f"   风格Token化: StatsTokenizer (3尺度×10统计token) + PatchTokenizer (3尺度×8=24)"
        )
        print(
            "   风格Token聚合: StyleCrossAttnAggregator (8 queries × 3 scales, 带残差)"
        )
        print(f"   风格Token混合: TokenMixer (2层Transformer, 46 tokens)")
        print("   风格Token池化: AttentionPooling (可学习加权)")
        print(
            f"   风格编码器: LatentEncoder ({self.token_dim} -> {self.style_olr_rank})"
        )
        print("   原型网络: PrototypeNetwork")
        print(f"   域分类头: Linear ({self.style_olr_rank} -> {self.n_domains})")
        print(f"   距离度量: {self.metric}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        support_domains: torch.Tensor,
        query_domains: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 联合 Stiefel 流形 + HSIC 独立性约束架构

        参数：
            support_images: 支持集图像 [N_s, 3, H, W]
            support_labels: 支持集标签 [N_s]
            query_images: 查询集图像 [N_q, 3, H, W]
            n_way: 类别数 (N-way)
            support_domains: 支持集域标签 [N_s]，取值范围 [0, 3]
            query_domains: 查询集域标签 [N_q]，取值范围 [0, 3]

        返回：
            outputs: 字典包含
                - logits: [N_q, n_way] 分类 logits
                - prototypes: [n_way, 640] 类别原型
                - z_c: [B_total, 640] 语义嵌入（已归一化）
                - z_s: [B_total, 640] 风格嵌入（已归一化）
                - u_c: [B_total, 256] 语义潜码
                - u_s: [B_total, 64] 风格潜码
                - domain_logits: [B_total, 4] 域分类 logits
                - support_domains: [N_s] 支持集域标签
                - query_domains: [N_q] 查询集域标签
        """
        # === 输入验证 ===
        N_s = support_images.shape[0]
        N_q = query_images.shape[0]

        if N_s == 0 or N_q == 0:
            raise ValueError(f"批次大小不能为 0：N_s={N_s}, N_q={N_q}")

        if n_way <= 0:
            raise ValueError(f"n_way 必须 > 0，得到: {n_way}")

        if (support_domains < 0).any() or (support_domains > 3).any():
            raise ValueError(
                f"支持集域标签必须在 [0, 3] 范围内，得到范围: "
                f"[{support_domains.min().item()}, {support_domains.max().item()}]"
            )

        if (query_domains < 0).any() or (query_domains > 3).any():
            raise ValueError(
                f"查询集域标签必须在 [0, 3] 范围内，得到范围: "
                f"[{query_domains.min().item()}, {query_domains.max().item()}]"
            )

        # === 步骤 1: 拼接支持集和查询集 ===
        all_images = torch.cat(
            [support_images, query_images], dim=0
        )  # [B_total, 3, H, W]
        B_total = N_s + N_q

        # === 步骤 2: 本征编码器提取金字塔特征 ===
        pyramid_features = self.backbone(all_images)  # {f1, f2, f3, f4}

        # === 步骤 3: 语义分支处理 ===
        f4 = pyramid_features["f4"]  # [B_total, 640, h, w]
        f4_pooled = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1)  # [B_total, 640]

        # 检测 NaN/Inf
        if torch.isnan(f4_pooled).any() or torch.isinf(f4_pooled).any():
            raise RuntimeError("检测到 NaN/Inf 在 f4_pooled 中")

        # 语义潜码编码
        u_c = self.sem_encoder(f4_pooled)  # [B_total, 256]

        # 语义投影到子空间
        z_c_raw = self.joint_basis.project_semantic(u_c)  # [B_total, 640]

        # L2 归一化
        z_c = F.normalize(z_c_raw, dim=1, eps=1e-8)  # [B_total, 640]

        # 检测 NaN/Inf
        if torch.isnan(z_c).any() or torch.isinf(z_c).any():
            raise RuntimeError("检测到 NaN/Inf 在语义嵌入 z_c 中")

        # === 旧语义分支代码（已注释） ===
        # u_c, A_c, z_c = self.sem_olr(f4_pooled)  # z_c: [B_total, 640]

        # === 步骤 4: 拆分语义嵌入 ===
        z_c_support = z_c[:N_s]  # [N_s, 640]
        z_c_query = z_c[N_s:]  # [N_q, 640]

        # === 步骤 5: 计算原型和分类 ===
        logits, prototypes = self.proto_head(
            z_c_support, support_labels, z_c_query, n_way
        )  # logits: [N_q, n_way], prototypes: [n_way, 640]

        # === 步骤 6: 风格分支处理（多尺度多因子Token化） ===
        # 第一步：构建风格输入，只包含 f1, f2, f3（不包含 f4）
        # f4 是高语义层，文档明确说域支只读早/中层
        style_pyramid = {
            "f1": pyramid_features["f1"].detach()
            if self.detach_style_inputs
            else pyramid_features["f1"],
            "f2": pyramid_features["f2"].detach()
            if self.detach_style_inputs
            else pyramid_features["f2"],
            "f3": pyramid_features["f3"].detach()
            if self.detach_style_inputs
            else pyramid_features["f3"],
        }

        # 第二步：统计 Token 化
        stats_tokens = self.stats_tokenizer(style_pyramid)  # [B_total, 22, 128]

        # 第三步：Patch Token 化 + Cross-Attn 聚合
        patch_tokens_list = self.patch_tokenizer(
            style_pyramid
        )  # List of 3 × [B, n_i, 128]
        patch_tokens = self.cross_attn_agg(patch_tokens_list)  # [B_total, 24, 128]

        # 第四步：合并所有风格 tokens
        all_style_tokens = torch.cat(
            [stats_tokens, patch_tokens], dim=1
        )  # [B_total, 46, 128]

        # 第五步：NaN/Inf 检测
        if torch.isnan(all_style_tokens).any() or torch.isinf(all_style_tokens).any():
            raise RuntimeError("检测到 NaN/Inf 在 all_style_tokens 中")

        # 第六步：Token 混合
        mixed_tokens = self.token_mixer(all_style_tokens)  # [B_total, 46, 128]

        # 第七步：注意力池化（替代 mean pooling）
        token_agg = self.attn_pool(mixed_tokens)  # [B_total, 128]

        # 第八步：NaN/Inf 检测
        if torch.isnan(token_agg).any() or torch.isinf(token_agg).any():
            raise RuntimeError("检测到 NaN/Inf 在 token_agg 中")

        # 风格潜码编码
        u_s = self.sty_encoder(token_agg)  # [B_total, 64]

        # 风格投影到子空间
        z_s_raw = self.joint_basis.project_style(u_s)  # [B_total, 640]

        # L2 归一化
        z_s = F.normalize(z_s_raw, dim=1, eps=1e-8)  # [B_total, 640]

        # 检测 NaN/Inf
        if torch.isnan(z_s).any() or torch.isinf(z_s).any():
            raise RuntimeError("检测到 NaN/Inf 在风格嵌入 z_s 中")

        # === 旧风格分支代码（已注释） ===
        # style_features = {
        #     "f1": pyramid_features["f1"],
        #     "f2": pyramid_features["f2"],
        #     "f3": pyramid_features["f3"]
        # }
        # u_s, z_s, domain_logits = self.style_branch(style_features)
        # # u_s: [B_total, 64], z_s: [B_total, 640], domain_logits: [B_total, 4]

        # === 步骤 7: 域分类（使用风格潜码 u_s） ===
        domain_logits = self.domain_head(u_s)  # [B_total, 4]

        # === 步骤 8: 返回完整字典 ===
        return {
            "logits": logits,  # [N_q, n_way]
            "prototypes": prototypes,  # [n_way, 640]
            "z_c": z_c,  # [B_total, 640]
            "z_s": z_s,  # [B_total, 640]
            "u_c": u_c,  # [B_total, 256]
            "u_s": u_s,  # [B_total, 64]
            "token_agg": token_agg,  # [B_total, 128] 风格Token聚合特征
            "domain_logits": domain_logits,  # [B_total, 4]
            "support_domains": support_domains,  # [N_s]
            "query_domains": query_domains,  # [N_q]
        }

    def set_mode(self, mode):
        """
        设置模型模式 (train/eval)
        """
        self.mode = mode
        if mode == "train":
            self.train()
        else:
            self.eval()

    def to_device(self, device):
        """
        将模型移动到指定设备
        """
        self.to(device)
        print(f"[OK] ExpB1Model 已移动到设备: {device}")
        return self

    def get_parameters(self):
        """
        获取模型参数

        Returns:
            list: 参数组列表，格式为 [{'params': [...]}]
        """
        # 返回所有参数
        return [{"params": self.parameters()}]

    def get_model_info(self):
        """
        获取模型信息字典
        """
        return {
            "model_name": "ExpB1Model_JointStiefel_HSIC",
            "backbone": "IntrinsicEncoder",
            "architecture": "联合 Stiefel 流形 + HSIC 独立性约束",
            "metric": self.metric,
        }


# 模型配置信息 - ExpB1Model 联合 Stiefel + HSIC 完整配置
MODEL_CONFIG = {
    "model_name": "ExpB1Model_JointStiefel_HSIC",
    "description": "联合 Stiefel 流形 + HSIC 独立性约束 - 语义-风格正交解耦域自适应小样本学习",
    "backbone": "IntrinsicEncoder",
    "architecture": "联合正交基底（语义+风格）",
    "key_features": [
        "金字塔特征提取",
        "联合正交矩阵 Q ∈ St(320, 640)",
        "潜码编码器（语义和风格）",
        "子空间投影和 L2 归一化",
        "HSIC 独立性约束",
        "域分类和风格解耦",
    ],
}
