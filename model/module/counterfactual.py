"""
反事实一致性模块 - AdaIN 风格干预 + 反事实前向传播

本模块实现双层 AdaIN 反事实干预，用于训练期的因果一致性约束。
核心功能：
1. feature_adain: 特征层 AdaIN 操作（实例级风格迁移）
2. counterfactual_forward: 双层干预的反事实前向传播

数学背景：
- AdaIN: output = σ_style * (content - μ_content) / (σ_content + ε) + μ_style
- 反事实路径: 通过风格交换生成反事实样本，检验语义不变性

作者：Kiro AI
日期：2024
"""

import torch
import torch.nn.functional as F
from typing import Dict


def feature_adain(
    content_feat: torch.Tensor, style_feat: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    特征层 AdaIN：用 style_feat 的逐通道均值/标准差替换 content_feat 的统计量。

    数学公式：
        output = σ_style * (content_feat - μ_content) / (σ_content + ε) + μ_style

    其中 μ, σ 均为逐样本-逐通道的空间统计量（instance-level）。

    Args:
        content_feat: 内容提供者的特征图 [B, C, H, W]
        style_feat: 风格提供者的特征图 [B, C, H, W]
                    形状必须与 content_feat 完全一致
        eps: 数值稳定性常数，默认 1e-6

    Returns:
        output: AdaIN 变换后的特征图 [B, C, H, W]
                空间结构保持 content_feat 的相对模式，
                统计量替换为 style_feat 的均值和标准差。

    注意：
        - 此函数不含任何可学习参数
        - 对 content_feat 和 style_feat 均可导（梯度可正常回流）
        - 不做任何 detach 操作（由调用方决定）
    """
    assert content_feat.shape == style_feat.shape, (
        f"content_feat 和 style_feat 形状必须一致，得到: {content_feat.shape} vs {style_feat.shape}"
    )

    mu_content = content_feat.mean(dim=[2, 3], keepdim=True)
    var_content = content_feat.var(dim=[2, 3], keepdim=True, unbiased=True)
    sigma_content = torch.sqrt(var_content + eps)

    mu_style = style_feat.mean(dim=[2, 3], keepdim=True)
    var_style = style_feat.var(dim=[2, 3], keepdim=True, unbiased=True)
    sigma_style = torch.sqrt(var_style + eps)

    normalized = (content_feat - mu_content) / sigma_content

    output = sigma_style * normalized + mu_style

    return output


def counterfactual_forward(
    model,
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor,
    perm: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    执行双层 AdaIN 反事实前向传播。

    计算流程：
        1. f1_cf = AdaIN(f1, f1[perm])          — 第一层风格干预
        2. f2_mid = model.backbone.layer2(f1_cf) — 重跑 layer2
        3. f2_cf = AdaIN(f2_mid, f2[perm])       — 第二层风格干预
        4. f3_cf = model.backbone.layer3(f2_cf)  — 重跑 layer3
        5. f4_cf = model.backbone.layer4(f3_cf)  — 重跑 layer4
        6. f4_cf → pool → sem_encoder → u_c_cf → project_semantic → z_c_cf
        7. {f1_cf, f2_cf, f3_cf} detach → style_pipeline → u_s_cf → z_s_cf

    Args:
        model: ExpB1Model 实例（访问 backbone.layer2/3/4、sem_encoder、
                sty_encoder、joint_basis、风格管线各模块）
        f1:     正常路径 backbone layer1 的输出 [B, 64, H1, W1]
        f2:     正常路径 backbone layer2 的输出 [B, 160, H2, W2]
        f3:     正常路径 backbone layer3 的输出 [B, 320, H3, W3]
        perm:   风格配对的排列索引 [B]，由 torch.randperm(B) 生成

    Returns:
        dict 包含：
            "z_c_cf":  反事实语义嵌入 [B, 640]（已 L2 归一化）
            "z_s_cf":  反事实风格嵌入 [B, 640]（已 L2 归一化）
            "u_s_cf":  反事实风格潜码 [B, rank_s]
            "perm":    使用的排列索引 [B]（原样返回，供损失函数使用）

    梯度流说明：
        - z_c_cf 的梯度可回流至：backbone.layer2/3/4、sem_encoder、A_c、f1（通过 AdaIN）
        - z_s_cf 的梯度可回流至：sty_encoder、style pipeline、A_s
        - z_s_cf 的梯度 **不会** 回流至 backbone（因为 detach_style_inputs）
        - backbone.stem 和 backbone.layer1 不在反事实计算图中
    """
    B = f1.shape[0]

    f1_cf = feature_adain(f1, f1[perm])

    f2_mid = model.backbone.layer2(f1_cf)

    f2_cf = feature_adain(f2_mid, f2[perm])

    f3_cf = model.backbone.layer3(f2_cf)

    f4_cf = model.backbone.layer4(f3_cf)

    f4_cf_pooled = F.adaptive_avg_pool2d(f4_cf, (1, 1)).flatten(1)
    u_c_cf = model.sem_encoder(f4_cf_pooled)
    z_c_cf_raw = model.joint_basis.project_semantic(u_c_cf)
    z_c_cf = F.normalize(z_c_cf_raw, dim=1, eps=1e-8)

    # === 风格分支：使用模型封装的风格编码管线 ===
    # 构建反事实风格金字塔（处理 detach 逻辑）
    if model.detach_style_inputs:
        style_pyramid_cf = {
            "f1": f1_cf.detach(),
            "f2": f2_cf.detach(),
            "f3": f3_cf.detach(),
        }
    else:
        style_pyramid_cf = {
            "f1": f1_cf,
            "f2": f2_cf,
            "f3": f3_cf,
        }

    # 调用模型的风格编码管线（复用代码，避免重复）
    style_outputs_cf = model.encode_style_features(
        style_pyramid_cf, return_intermediates=False
    )
    z_s_cf = style_outputs_cf["z_s"]  # [B, 640]
    u_s_cf = style_outputs_cf["u_s"]  # [B, rank_s]

    return {
        "z_c_cf": z_c_cf,  # [B, 640]
        "z_s_cf": z_s_cf,  # [B, 640]
        "u_s_cf": u_s_cf,  # [B, rank_s]
        "perm": perm,  # [B]
    }
