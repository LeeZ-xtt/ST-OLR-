"""
ST-OLR 损失函数模块

实现所有ST-OLR系统所需的损失函数：
1. 域分类损失
2. HSIC 独立性损失
3. 总损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def domain_classification_loss(
    domain_logits: torch.Tensor, domain_labels: torch.Tensor
) -> torch.Tensor:
    """
    域分类交叉熵损失

    Args:
        domain_logits: 域分类logits [B, 4]
        domain_labels: 域标签 [B]，取值范围 [0, 3]

    Returns:
        loss: 标量损失值

    Raises:
        ValueError: 当域标签包含无效值时
    """
    # 验证域标签范围
    if (domain_labels < 0).any() or (domain_labels > 3).any():
        raise ValueError(
            f"域标签必须在 [0, 3] 范围内，得到: "
            f"min={domain_labels.min().item()}, max={domain_labels.max().item()}"
        )

    # 交叉熵损失 + label smoothing 防止域分类头过拟合
    loss = F.cross_entropy(domain_logits, domain_labels, label_smoothing=0.05)

    return loss


class HSICIndependenceLoss(nn.Module):
    """
    基于 Hilbert-Schmidt 独立性准则 (HSIC) 的独立性损失

    使用 RBF 核测量两个特征表示之间的非线性统计依赖性。
    HSIC 值越小表示两个表示越独立。

    算法流程：
    1. 计算成对平方距离矩阵 (使用 torch.cdist)
    2. 使用 median heuristic 自适应确定 RBF 核带宽
    3. 计算 RBF 核矩阵：K = exp(-d^2 / (2*sigma^2))
    4. 中心化核矩阵：Kc = H K H，其中 H = I - 1/B
    5. 计算 HSIC 统计量：tr(Kxc * Kyc) / (B-1)^2

    数学定义：
        HSIC(X, Y) = (1/(B-1)^2) * tr(Kxc * Kyc)
        其中 Kxc 和 Kyc 是中心化的核矩阵

    属性：
        - HSIC >= 0 (非负性)
        - HSIC = 0 当且仅当 X 和 Y 独立 (在总体层面)
        - 梯度同时流向两个输入 (双向对称)

    参考文献：
        Gretton et al. "Measuring Statistical Dependence with
        Hilbert-Schmidt Norms." ALT 2005.
    """

    def __init__(self):
        """初始化 HSIC 独立性损失"""
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算 HSIC 独立性损失

        Args:
            x: 第一组特征 [B, d_x]，可以是任意维度的特征向量，不需要归一化
            y: 第二组特征 [B, d_y]，维度可以与 x 相同或不同，不需要归一化

        Returns:
            hsic: HSIC 统计量 (标量)，值越小表示越独立

        Raises:
            ValueError: 当批次大小 < 2 时
            RuntimeWarning: 当输入包含 NaN/Inf 时

        形状说明：
            HSIC 使用核方法，核矩阵是 B×B 的，与输入维度无关。
            x: [B, d_x] -> dx: [B, B] -> Kx: [B, B] -> Kxc: [B, B]
            y: [B, d_y] -> dy: [B, B] -> Ky: [B, B] -> Kyc: [B, B]
            hsic: 标量
        """
        B = x.shape[0]

        if B < 2:
            raise ValueError(f"HSIC 需要至少 2 个样本进行统计估计，得到批次大小: {B}")

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("检测到 NaN/Inf 在输入 x 中，返回零")
            return torch.zeros(1, device=x.device)
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.warning("检测到 NaN/Inf 在输入 y 中，返回零")
            return torch.zeros(1, device=y.device)

        dx = torch.cdist(x, x).pow(2)
        dy = torch.cdist(y, y).pow(2)

        mask = torch.triu(
            torch.ones(B, B, dtype=torch.bool, device=x.device), diagonal=1
        )
        sigma_x = dx[mask].median().detach().clamp(min=1e-2, max=1e4)
        sigma_y = dy[mask].median().detach().clamp(min=1e-2, max=1e4)

        Kx = (-dx / (2 * sigma_x)).exp()
        Ky = (-dy / (2 * sigma_y)).exp()

        Kx_row_mean = Kx.mean(dim=1, keepdim=True)
        Kx_col_mean = Kx.mean(dim=0, keepdim=True)
        Kx_total_mean = Kx.mean()
        Kxc = Kx - Kx_row_mean - Kx_col_mean + Kx_total_mean

        Ky_row_mean = Ky.mean(dim=1, keepdim=True)
        Ky_col_mean = Ky.mean(dim=0, keepdim=True)
        Ky_total_mean = Ky.mean()
        Kyc = Ky - Ky_row_mean - Ky_col_mean + Ky_total_mean

        hsic_xy = (Kxc * Kyc).sum()
        hsic_xx = (Kxc * Kxc).sum()
        hsic_yy = (Kyc * Kyc).sum()
        eps = 1e-8
        nhsic = hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + eps)

        return nhsic


_HSIC_LOSS = HSICIndependenceLoss()


def counterfactual_semantic_loss(
    z_c: torch.Tensor,
    z_c_cf: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 10.0,
) -> torch.Tensor:
    """
    反事实语义不变性损失 L_cf_sema

    确保语义嵌入 z_c 对风格干预不变：
    对风格做 do-干预后，分类概率分布应保持不变。

    数学公式：
        p_orig_i = softmax(-temperature * ||z_c[i] - p_k||^2)  ∀k
        p_cf_i   = softmax(-temperature * ||z_c_cf[i] - p_k||^2)  ∀k
        L = mean_i KL(p_orig_i || p_cf_i)

    Args:
        z_c:         正常路径语义嵌入 [B, 640]，已 L2 归一化
        z_c_cf:      反事实路径语义嵌入 [B, 640]，已 L2 归一化
        prototypes:  类别原型 [n_way, 640]，来自正常路径的 proto_head 输出
        temperature: softmax 温度参数（与原型网络一致，默认 10.0）

    Returns:
        loss: 标量。值越小表示语义对风格越不变。

    梯度流：
        - 原始分布 p_orig 被 detach，梯度不回流至正常路径的 z_c
        - 梯度仅通过 p_cf 回流至反事实路径的 z_c_cf
        - 原型 prototypes 被 detach，避免循环梯度
    """
    dists_orig = -temperature * torch.cdist(z_c, prototypes.detach()).pow(2)
    p_orig = F.softmax(dists_orig, dim=1).detach()

    dists_cf = -temperature * torch.cdist(z_c_cf, prototypes.detach()).pow(2)
    log_p_cf = F.log_softmax(dists_cf, dim=1)

    loss = F.kl_div(log_p_cf, p_orig, reduction="batchmean")

    return loss


def style_following_loss(
    z_s_cf: torch.Tensor,
    z_s: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    风格跟随损失 L_sf

    使反事实路径的风格嵌入 z_s_cf 跟随风格提供者的 z_s。
    这是修复 A_s 梯度断流的核心损失。

    数学公式：
        L = mean_i (1 - cos_sim(z_s_cf[i], z_s[perm[i]]))

    Args:
        z_s_cf:  反事实路径风格嵌入 [B, 640]，已 L2 归一化
        z_s:     正常路径风格嵌入 [B, 640]，已 L2 归一化
        perm:    风格配对排列索引 [B]（LongTensor）

    Returns:
        loss: 标量。值域 [0, 2]，0 表示完全对齐。

    梯度流：
        - z_s[perm] 被 detach，梯度不回流至正常路径的 z_s
        - 梯度仅通过 z_s_cf 回流至：A_s（关键！）、sty_encoder、style pipeline
    """
    z_s_target = z_s[perm].detach()

    cos_sim = F.cosine_similarity(z_s_cf, z_s_target, dim=1)

    loss = (1.0 - cos_sim).mean()

    return loss


def total_loss(
    outputs: Dict[str, torch.Tensor],
    query_labels: torch.Tensor,
    cf_outputs: Optional[Dict[str, torch.Tensor]] = None,
    current_epoch: int = 1,
    config=None,
) -> Dict[str, torch.Tensor]:
    """
    计算总损失（简化接口版本）

    自动从 config 读取损失权重并计算反事实 warmup 系数。

    Args:
        outputs: 模型输出字典，包含:
            - logits: [N_q, n_way] 分类logits
            - domain_logits: [B_total, 4] 域分类logits
            - z_c: [B_total, 640] 语义嵌入（已归一化）
            - z_s: [B_total, 640] 风格嵌入（已归一化）
            - u_c: [B_total, rank_c] 语义潜码
            - u_s: [B_total, rank_s] 风格潜码
            - support_domains: [N_s] 支持集域标签
            - query_domains: [N_q] 查询集域标签
            - prototypes: [n_way, 640] 类别原型
        query_labels: 查询集标签 [N_q]
        cf_outputs: 反事实前向输出字典（训练期传入，推理期为None），包含:
            - z_c_cf: [B, 640] 反事实语义嵌入
            - z_s_cf: [B, 640] 反事实风格嵌入
            - perm: [B] 排列索引
        current_epoch: 当前训练 epoch（1-indexed），用于计算反事实 warmup
        config: 配置对象（默认使用全局 Config）

    Returns:
        losses: 包含所有损失项的字典
            - total: 总损失
            - cls: 分类损失
            - domain: 域分类损失
            - hsic: HSIC 独立性损失
            - cf_sema: 反事实语义损失
            - sf: 风格跟随损失
    """
    # 使用全局 Config 如果未提供
    if config is None:
        from config import Config

        config = Config

    # 计算反事实 warmup 系数
    cf_is_active = current_epoch >= config.cf_start_epoch
    if cf_is_active:
        epochs_since_start = current_epoch - config.cf_start_epoch
        cf_multiplier = min(1.0, epochs_since_start / config.cf_rampup_epochs)
    else:
        cf_multiplier = 0.0

    # 计算各项损失
    loss_cls = F.cross_entropy(outputs["logits"], query_labels)

    all_domains = torch.cat(
        [outputs["support_domains"], outputs["query_domains"]], dim=0
    )
    loss_domain = domain_classification_loss(outputs["domain_logits"], all_domains)

    B_total = outputs["u_c"].shape[0]
    if B_total >= 2:
        loss_hsic = _HSIC_LOSS(outputs["u_c"], outputs["u_s"])
    else:
        loss_hsic = torch.tensor(0.0, device=outputs["u_c"].device)

    # 反事实损失（仅在训练期且 warmup 后计算）
    if cf_outputs is not None and cf_is_active:
        loss_cf_sema = counterfactual_semantic_loss(
            z_c=outputs["z_c"],
            z_c_cf=cf_outputs["z_c_cf"],
            prototypes=outputs["prototypes"],
            temperature=config.proto_temperature,  # 从配置读取温度参数
        )
        loss_sf = style_following_loss(
            z_s_cf=cf_outputs["z_s_cf"],
            z_s=outputs["z_s"],
            perm=cf_outputs["perm"],
        )
    else:
        device = outputs["u_c"].device
        loss_cf_sema = torch.tensor(0.0, device=device)
        loss_sf = torch.tensor(0.0, device=device)

    # 加权求和（反事实损失自动应用 warmup 系数）
    total = (
        config.loss_weight_cls * loss_cls
        + config.loss_weight_domain * loss_domain
        + config.loss_weight_hsic * loss_hsic
        + config.loss_weight_cf_sema * cf_multiplier * loss_cf_sema
        + config.loss_weight_sf * cf_multiplier * loss_sf
    )

    return {
        "total": total,
        "cls": loss_cls,
        "domain": loss_domain,
        "hsic": loss_hsic,
        "cf_sema": loss_cf_sema,
        "sf": loss_sf,
    }
