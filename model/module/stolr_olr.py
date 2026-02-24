"""
ST-OLR 正交低秩投影模块

实现正交低秩（OLR）投影，将高维特征分解为低秩潜码和正交投影。
通过QR分解保证硬正交性，使用L2归一化保证嵌入空间的几何一致性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OLRProjector(nn.Module):
    """
    正交低秩投影器
    
    将高维特征 x 分解为：
    1. 低秩潜码 u = MLP(x)
    2. 正交基 A (通过QR分解保证 A^T @ A = I)
    3. 投影嵌入 z = normalize(Linear(u @ A.T))
    
    Args:
        in_dim: 输入特征维度
        rank: 低秩潜码维度
        mlp_hidden: MLP隐藏层维度
    """
    
    def __init__(self, in_dim: int, rank: int, mlp_hidden: int):
        super().__init__()
        
        self.in_dim = in_dim
        self.rank = rank
        self.mlp_hidden = mlp_hidden
        
        # 低秩编码MLP: in_dim → mlp_hidden → rank
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, rank)
        )
        
        # 正交基初始化矩阵 [in_dim, rank]
        self.A_raw = nn.Parameter(torch.empty(in_dim, rank))
        
        # 投影头: in_dim → 640
        self.proj_head = nn.Linear(in_dim, 640)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # MLP使用Kaiming初始化（适配GELU）
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 正交基使用正交初始化
        nn.init.orthogonal_(self.A_raw)
        
        # 投影头使用Kaiming初始化
        nn.init.kaiming_normal_(self.proj_head.weight, mode='fan_in', nonlinearity='relu')
        if self.proj_head.bias is not None:
            nn.init.zeros_(self.proj_head.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, in_dim]
        
        Returns:
            u: 低秩潜码 [B, rank]
            A: 正交基 [in_dim, rank]，满足 A^T @ A = I
            z: 投影嵌入 [B, 640]，L2归一化
        """
        # 步骤1: 低秩编码
        u = self.mlp(x)  # [B, in_dim] → [B, rank]
        
        # 步骤2: 正交基生成（每次前向都重新QR分解）
        Q, R = torch.linalg.qr(self.A_raw)  # Q: [in_dim, rank]
        A = Q  # 正交基
        
        # 步骤3: 投影到原始维度
        z_proj = u @ A.T  # [B, rank] @ [rank, in_dim] → [B, in_dim]
        
        # 步骤4: 升维到640维
        z = self.proj_head(z_proj)  # [B, in_dim] → [B, 640]
        
        # 步骤5: L2归一化
        z = F.normalize(z, p=2, dim=1)  # [B, 640]
        
        return u, A, z
    
    def get_orthogonality_error(self) -> torch.Tensor:
        """
        计算正交性误差 ||A^T @ A - I||_F
        
        Returns:
            error: 标量张量，表示正交性误差
        """
        Q, R = torch.linalg.qr(self.A_raw)
        identity = Q.T @ Q  # [rank, rank]
        expected_identity = torch.eye(self.rank, device=Q.device)
        error = torch.norm(identity - expected_identity, p='fro')
        return error
