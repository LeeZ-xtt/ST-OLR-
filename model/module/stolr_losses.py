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


def total_loss(
    outputs: Dict[str, torch.Tensor],
    query_labels: torch.Tensor,
    loss_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    计算总损失

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
        query_labels: 查询集标签 [N_q]
        loss_weights: 损失权重字典，包含:
            - cls: 分类损失权重
            - domain: 域分类损失权重
            - hsic: HSIC 独立性损失权重

    Returns:
        losses: 包含所有损失项的字典
            - total: 总损失
            - cls: 分类损失
            - domain: 域分类损失
            - hsic: HSIC 独立性损失
    """
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

    total = (
        loss_weights["cls"] * loss_cls
        + loss_weights["domain"] * loss_domain
        + loss_weights["hsic"] * loss_hsic
    )

    return {"total": total, "cls": loss_cls, "domain": loss_domain, "hsic": loss_hsic}
