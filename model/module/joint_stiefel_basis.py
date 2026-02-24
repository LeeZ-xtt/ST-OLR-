"""
联合 Stiefel 流形正交基底模块

本模块实现联合正交矩阵 Q ∈ St(320, 640)，通过列块切片提供语义和风格子空间的正交基底。
核心优势：A_c^T A_s = 0 成为结构恒等式，无需软约束损失。

数学背景：
- Stiefel 流形 St(n, m) = {Q ∈ R^{m×n} : Q^T Q = I_n}
- 联合矩阵 Q ∈ R^{640×320}，满足 Q^T Q = I_320
- 语义基底 A_c = Q[:, :256]，风格基底 A_s = Q[:, 256:320]
- 子空间正交性：A_c^T A_s = 0（由 Q 的分块结构保证）

作者：Kiro AI
日期：2024
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import orthogonal
from typing import Optional


class JointStiefelBasis(nn.Module):
    """
    联合 Stiefel 流形正交基底
    
    本类实现一个联合正交矩阵 Q ∈ R^{ambient_dim × total_rank}，其中 total_rank = rank_c + rank_s。
    通过列块切片，Q 提供两个正交子空间的标准正交基：
    - 语义子空间：A_c = Q[:, :rank_c]
    - 风格子空间：A_s = Q[:, rank_c:]
    
    关键性质：
    1. Q^T Q = I_{total_rank}（联合正交性）
    2. A_c^T A_c = I_{rank_c}（语义子空间内部正交）
    3. A_s^T A_s = I_{rank_s}（风格子空间内部正交）
    4. A_c^T A_s = 0（子空间间正交性，结构恒等式）
    
    使用 PyTorch 的 parametrizations.orthogonal 自动维护正交约束，
    在参数更新时自动将梯度投影到 Stiefel 流形的切空间。
    
    参数：
        ambient_dim (int): 环境空间维度，默认 640
        rank_c (int): 语义子空间秩，默认 256
        rank_s (int): 风格子空间秩，默认 64
    
    属性：
        Q (Tensor): 联合正交矩阵 [ambient_dim, total_rank]
    
    示例：
        >>> basis = JointStiefelBasis(ambient_dim=640, rank_c=256, rank_s=64)
        >>> u_c = torch.randn(16, 256)  # 语义潜码
        >>> u_s = torch.randn(16, 64)   # 风格潜码
        >>> z_c = basis.project_semantic(u_c)  # [16, 640]
        >>> z_s = basis.project_style(u_s)     # [16, 640]
        >>> 
        >>> # 验证正交性
        >>> Q = basis.Q
        >>> identity = Q.T @ Q
        >>> print(torch.norm(identity - torch.eye(320)))  # 应接近 0
    """
    
    def __init__(
        self,
        ambient_dim: int = 640,
        rank_c: int = 256,
        rank_s: int = 64
    ):
        """
        初始化联合正交基底
        
        参数：
            ambient_dim: 环境空间维度（嵌入维度），必须 >= rank_c + rank_s
            rank_c: 语义子空间秩（语义潜码维度）
            rank_s: 风格子空间秩（风格潜码维度）
        
        异常：
            ValueError: 当 ambient_dim < rank_c + rank_s 时
            ValueError: 当任何维度参数 <= 0 时
        """
        super().__init__()
        
        # 输入验证
        if ambient_dim <= 0:
            raise ValueError(f"环境空间维度必须 > 0，得到: {ambient_dim}")
        if rank_c <= 0:
            raise ValueError(f"语义子空间秩必须 > 0，得到: {rank_c}")
        if rank_s <= 0:
            raise ValueError(f"风格子空间秩必须 > 0，得到: {rank_s}")
        
        total_rank = rank_c + rank_s
        if ambient_dim < total_rank:
            raise ValueError(
                f"环境空间维度 ({ambient_dim}) 必须 >= 总秩 ({total_rank})"
            )
        
        # 存储维度参数
        self.ambient_dim = ambient_dim
        self.rank_c = rank_c
        self.rank_s = rank_s
        self.total_rank = total_rank
        
        # 创建底层线性层（无偏置）
        # 权重形状：[ambient_dim, total_rank]
        # 由于 m > n (640 > 320)，参数化将产生列正交矩阵
        self._basis = nn.Linear(total_rank, ambient_dim, bias=False)
        
        # 应用正交参数化
        # 使用 Householder 反射实现，数值稳定性优于 Gram-Schmidt
        # 自动维护 Q^T Q = I_{total_rank} 约束
        orthogonal(self._basis, 'weight')
        
    @property
    def Q(self) -> Tensor:
        """
        返回联合正交矩阵
        
        返回：
            Q: 联合正交矩阵 [ambient_dim, total_rank]，满足 Q^T Q = I_{total_rank}
        
        注意：
            - Q 的列向量构成 R^{ambient_dim} 的 total_rank 维子空间的标准正交基
            - 正交性由 parametrizations.orthogonal 自动维护
            - 正交性误差通常 < 1e-6
        """
        return self._basis.weight  # [ambient_dim, total_rank]
    
    def project_semantic(self, u_c: Tensor) -> Tensor:
        """
        将语义潜码投影到语义子空间
        
        计算公式：z_c = u_c @ A_c^T，其中 A_c = Q[:, :rank_c]
        
        参数：
            u_c: 语义潜码 [B, rank_c]
        
        返回：
            z_c_raw: 语义嵌入 [B, ambient_dim]（未归一化）
        
        异常：
            RuntimeError: 当 u_c 维度与 rank_c 不匹配时
            RuntimeError: 当 u_c 包含 NaN 或 Inf 时
        
        注意：
            - 返回值通常需要 L2 归一化：z_c = F.normalize(z_c_raw, dim=1)
            - A_c 满足 A_c^T A_c = I_{rank_c}（由 Q 的正交性保证）
            - 投影操作保持在语义子空间内
        
        示例：
            >>> basis = JointStiefelBasis(640, 256, 64)
            >>> u_c = torch.randn(16, 256)
            >>> z_c_raw = basis.project_semantic(u_c)  # [16, 640]
            >>> z_c = F.normalize(z_c_raw, dim=1)  # L2 归一化
        """
        # 输入验证
        if u_c.dim() != 2:
            raise RuntimeError(
                f"语义潜码必须是 2D 张量 [B, rank_c]，得到形状: {u_c.shape}"
            )
        
        if u_c.shape[1] != self.rank_c:
            raise RuntimeError(
                f"语义潜码维度 ({u_c.shape[1]}) 与基底秩 ({self.rank_c}) 不匹配"
            )
        
        # NaN/Inf 检测
        if torch.isnan(u_c).any() or torch.isinf(u_c).any():
            raise RuntimeError("语义潜码包含 NaN 或 Inf 值")
        
        # 提取语义基底：A_c = Q[:, :rank_c]
        A_c = self.Q[:, :self.rank_c]  # [ambient_dim, rank_c] = [640, 256]
        
        # 投影：z_c = u_c @ A_c^T
        z_c_raw = u_c @ A_c.T  # [B, rank_c] @ [rank_c, ambient_dim] = [B, ambient_dim]
        
        return z_c_raw  # [B, 640]
    
    def project_style(self, u_s: Tensor) -> Tensor:
        """
        将风格潜码投影到风格子空间
        
        计算公式：z_s = u_s @ A_s^T，其中 A_s = Q[:, rank_c:]
        
        参数：
            u_s: 风格潜码 [B, rank_s]
        
        返回：
            z_s_raw: 风格嵌入 [B, ambient_dim]（未归一化）
        
        异常：
            RuntimeError: 当 u_s 维度与 rank_s 不匹配时
            RuntimeError: 当 u_s 包含 NaN 或 Inf 时
        
        注意：
            - 返回值通常需要 L2 归一化：z_s = F.normalize(z_s_raw, dim=1)
            - A_s 满足 A_s^T A_s = I_{rank_s}（由 Q 的正交性保证）
            - A_c^T A_s = 0（子空间正交性，结构恒等式）
            - 投影操作保持在风格子空间内
        
        示例：
            >>> basis = JointStiefelBasis(640, 256, 64)
            >>> u_s = torch.randn(16, 64)
            >>> z_s_raw = basis.project_style(u_s)  # [16, 640]
            >>> z_s = F.normalize(z_s_raw, dim=1)  # L2 归一化
        """
        # 输入验证
        if u_s.dim() != 2:
            raise RuntimeError(
                f"风格潜码必须是 2D 张量 [B, rank_s]，得到形状: {u_s.shape}"
            )
        
        if u_s.shape[1] != self.rank_s:
            raise RuntimeError(
                f"风格潜码维度 ({u_s.shape[1]}) 与基底秩 ({self.rank_s}) 不匹配"
            )
        
        # NaN/Inf 检测
        if torch.isnan(u_s).any() or torch.isinf(u_s).any():
            raise RuntimeError("风格潜码包含 NaN 或 Inf 值")
        
        # 提取风格基底：A_s = Q[:, rank_c:]
        A_s = self.Q[:, self.rank_c:]  # [ambient_dim, rank_s] = [640, 64]
        
        # 投影：z_s = u_s @ A_s^T
        z_s_raw = u_s @ A_s.T  # [B, rank_s] @ [rank_s, ambient_dim] = [B, ambient_dim]
        
        return z_s_raw  # [B, 640]
    
    def get_semantic_basis(self) -> Tensor:
        """
        获取语义子空间的标准正交基
        
        返回：
            A_c: 语义基底 [ambient_dim, rank_c]，满足 A_c^T A_c = I_{rank_c}
        """
        return self.Q[:, :self.rank_c]
    
    def get_style_basis(self) -> Tensor:
        """
        获取风格子空间的标准正交基
        
        返回：
            A_s: 风格基底 [ambient_dim, rank_s]，满足 A_s^T A_s = I_{rank_s}
        """
        return self.Q[:, self.rank_c:]
    
    def verify_orthogonality(self, atol: float = 1e-5) -> dict:
        """
        验证正交性约束（用于调试和测试）
        
        参数：
            atol: 绝对容差阈值
        
        返回：
            dict: 包含各项正交性误差的字典
                - 'joint_orthogonality': ||Q^T Q - I||_F
                - 'semantic_orthogonality': ||A_c^T A_c - I||_F
                - 'style_orthogonality': ||A_s^T A_s - I||_F
                - 'subspace_orthogonality': ||A_c^T A_s||_F
                - 'all_satisfied': 是否所有约束都满足
        
        示例：
            >>> basis = JointStiefelBasis(640, 256, 64)
            >>> errors = basis.verify_orthogonality()
            >>> print(f"联合正交性误差: {errors['joint_orthogonality']:.2e}")
            >>> print(f"子空间正交性误差: {errors['subspace_orthogonality']:.2e}")
        """
        Q = self.Q
        A_c = self.get_semantic_basis()
        A_s = self.get_style_basis()
        
        # 计算各项误差
        joint_error = torch.norm(
            Q.T @ Q - torch.eye(self.total_rank, device=Q.device)
        ).item()
        
        semantic_error = torch.norm(
            A_c.T @ A_c - torch.eye(self.rank_c, device=A_c.device)
        ).item()
        
        style_error = torch.norm(
            A_s.T @ A_s - torch.eye(self.rank_s, device=A_s.device)
        ).item()
        
        subspace_error = torch.norm(A_c.T @ A_s).item()
        
        # 检查是否所有约束都满足
        all_satisfied = (
            joint_error < atol and
            semantic_error < atol and
            style_error < atol and
            subspace_error < atol
        )
        
        return {
            'joint_orthogonality': joint_error,
            'semantic_orthogonality': semantic_error,
            'style_orthogonality': style_error,
            'subspace_orthogonality': subspace_error,
            'all_satisfied': all_satisfied
        }
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"ambient_dim={self.ambient_dim}, "
            f"rank_c={self.rank_c}, "
            f"rank_s={self.rank_s}, "
            f"total_rank={self.total_rank}"
        )


if __name__ == "__main__":
    # 简单测试
    print("=" * 60)
    print("JointStiefelBasis 模块测试")
    print("=" * 60)
    
    # 创建实例
    basis = JointStiefelBasis(ambient_dim=640, rank_c=256, rank_s=64)
    print(f"\n模块信息: {basis}")
    
    # 验证正交性
    errors = basis.verify_orthogonality()
    print(f"\n正交性验证:")
    print(f"  联合正交性误差: {errors['joint_orthogonality']:.2e}")
    print(f"  语义子空间正交性误差: {errors['semantic_orthogonality']:.2e}")
    print(f"  风格子空间正交性误差: {errors['style_orthogonality']:.2e}")
    print(f"  子空间间正交性误差: {errors['subspace_orthogonality']:.2e}")
    print(f"  所有约束满足: {errors['all_satisfied']}")
    
    # 测试投影
    batch_size = 16
    u_c = torch.randn(batch_size, 256)
    u_s = torch.randn(batch_size, 64)
    
    z_c_raw = basis.project_semantic(u_c)
    z_s_raw = basis.project_style(u_s)
    
    print(f"\n投影测试:")
    print(f"  语义潜码形状: {u_c.shape} -> 语义嵌入形状: {z_c_raw.shape}")
    print(f"  风格潜码形状: {u_s.shape} -> 风格嵌入形状: {z_s_raw.shape}")
    
    # L2 归一化
    import torch.nn.functional as F
    z_c = F.normalize(z_c_raw, dim=1)
    z_s = F.normalize(z_s_raw, dim=1)
    
    print(f"\n归一化后:")
    print(f"  z_c 平均范数: {torch.norm(z_c, dim=1).mean():.6f}")
    print(f"  z_s 平均范数: {torch.norm(z_s, dim=1).mean():.6f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
