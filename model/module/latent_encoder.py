"""
潜码编码器模块

该模块实现了将高维特征编码为低秩潜码表示的 MLP 编码器。
用于语义和风格特征的潜码编码。

作者：ST-OLR 项目组
日期：2024
"""

import torch
import torch.nn as nn
from typing import Optional


class LatentEncoder(nn.Module):
    """
    潜码编码器

    使用 MLP 结构将高维特征编码为低秩潜码表示：
    Linear(in_dim, hidden) → LayerNorm → GELU → Linear(hidden, rank)

    其中隐藏层维度 hidden = in_dim * 2

    使用场景：
    - 语义编码器：LatentEncoder(640, 256)
    - 风格编码器：LatentEncoder(128, 64)

    Args:
        in_dim: 输入特征维度
        rank: 输出潜码维度（子空间秩）

    示例：
        >>> # 创建语义编码器
        >>> sem_encoder = LatentEncoder(in_dim=640, rank=256)
        >>> x = torch.randn(16, 640)  # [B, 640]
        >>> u_c = sem_encoder(x)  # [B, 256]
        >>> print(u_c.shape)
        torch.Size([16, 256])

        >>> # 创建风格编码器
        >>> sty_encoder = LatentEncoder(in_dim=128, rank=64)
        >>> x = torch.randn(16, 128)  # [B, 128]
        >>> u_s = sty_encoder(x)  # [B, 64]
        >>> print(u_s.shape)
        torch.Size([16, 64])
    """

    def __init__(self, in_dim: int, rank: int):
        """
        初始化潜码编码器

        Args:
            in_dim: 输入特征维度（语义：640，风格：128）
            rank: 输出潜码维度（语义：256，风格：64）

        Raises:
            ValueError: 如果 in_dim 或 rank 不是正整数
        """
        super().__init__()

        # 参数验证
        if in_dim <= 0:
            raise ValueError(f"输入维度必须为正整数，得到: {in_dim}")
        if rank <= 0:
            raise ValueError(f"潜码秩必须为正整数，得到: {rank}")

        self.in_dim = in_dim
        self.rank = rank
        self.hidden_dim = in_dim * 2  # 隐藏层维度为输入维度的 2 倍

        # MLP 结构：Linear → LayerNorm → GELU → Dropout → Linear
        # 语义实例：[B, 640] -> [B, 1280] -> [B, 1280] -> [B, 1280] -> [B, 256]
        # 风格实例：[B, 128] -> [B, 256] -> [B, 256] -> [B, 256] -> [B, 64]
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),  # [B, in_dim] -> [B, hidden_dim]
            nn.LayerNorm(self.hidden_dim),  # [B, hidden_dim] -> [B, hidden_dim]
            nn.GELU(),  # [B, hidden_dim] -> [B, hidden_dim]
            nn.Dropout(0.1),  # 防止 episodic 训练中的过拟合
            nn.Linear(self.hidden_dim, rank),  # [B, hidden_dim] -> [B, rank]
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化网络权重

        使用分层初始化策略：
        - 中间层（GELU 前）：Xavier uniform（GELU 增益近似线性）
        - 最后一层（无激活）：Xavier uniform（线性输出）

        理论依据：
        - GELU ≈ x·Φ(x)，实践中 Xavier 初始化最稳定
        - 最后一层无激活函数，不应使用 Kaiming-ReLU 增益

        LayerNorm 的参数使用默认初始化（gamma=1, beta=0）。
        """
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                if i == len(self.net) - 1:
                    # 最后一层（无激活）：Xavier uniform
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # 中间层（GELU 前）：Xavier uniform（GELU 近似线性增益）
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征张量 [B, in_dim]
               其中 B 是批次大小

        Returns:
            u: 潜码张量 [B, rank]

        Raises:
            ValueError: 如果输入形状不正确
            RuntimeError: 如果输入包含 NaN 或 Inf

        示例：
            >>> encoder = LatentEncoder(640, 256)
            >>> x = torch.randn(16, 640)
            >>> u = encoder(x)
            >>> print(u.shape)
            torch.Size([16, 256])
        """
        # 输入验证：检查形状
        if x.dim() != 2:
            raise ValueError(f"输入张量必须是 2 维 [B, in_dim]，得到形状: {x.shape}")

        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"输入特征维度不匹配：期望 {self.in_dim}，得到 {x.shape[1]}"
            )

        # 输入验证：检测 NaN/Inf
        if torch.isnan(x).any():
            raise RuntimeError("输入张量包含 NaN 值")
        if torch.isinf(x).any():
            raise RuntimeError("输入张量包含 Inf 值")

        # 前向传播
        u = self.net(x)  # [B, in_dim] -> [B, rank]

        return u

    def __repr__(self) -> str:
        """返回模块的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, "
            f"rank={self.rank}, "
            f"hidden_dim={self.hidden_dim})"
        )


if __name__ == "__main__":
    """测试代码"""
    print("=" * 60)
    print("测试 LatentEncoder 模块")
    print("=" * 60)

    # 测试 1: 语义编码器
    print("\n测试 1: 语义编码器 (640 -> 256)")
    sem_encoder = LatentEncoder(in_dim=640, rank=256)
    print(f"模型结构: {sem_encoder}")

    x_sem = torch.randn(16, 640)
    u_c = sem_encoder(x_sem)
    print(f"输入形状: {x_sem.shape}")
    print(f"输出形状: {u_c.shape}")
    assert u_c.shape == (16, 256), "语义编码器输出形状错误"
    print("✓ 语义编码器测试通过")

    # 测试 2: 风格编码器
    print("\n测试 2: 风格编码器 (128 -> 64)")
    sty_encoder = LatentEncoder(in_dim=128, rank=64)
    print(f"模型结构: {sty_encoder}")

    x_sty = torch.randn(16, 128)
    u_s = sty_encoder(x_sty)
    print(f"输入形状: {x_sty.shape}")
    print(f"输出形状: {u_s.shape}")
    assert u_s.shape == (16, 64), "风格编码器输出形状错误"
    print("✓ 风格编码器测试通过")

    # 测试 3: 批次大小变化
    print("\n测试 3: 不同批次大小")
    for batch_size in [1, 8, 32, 64]:
        x = torch.randn(batch_size, 640)
        u = sem_encoder(x)
        assert u.shape == (batch_size, 256), f"批次大小 {batch_size} 测试失败"
        print(f"  批次大小 {batch_size}: {x.shape} -> {u.shape} ✓")
    print("✓ 批次大小变化测试通过")

    # 测试 4: 梯度流
    print("\n测试 4: 梯度反向传播")
    x = torch.randn(16, 640, requires_grad=True)
    u = sem_encoder(x)
    loss = u.sum()
    loss.backward()
    assert x.grad is not None, "输入梯度为 None"
    assert sem_encoder.net[0].weight.grad is not None, "第一层权重梯度为 None"
    print(f"输入梯度范数: {x.grad.norm().item():.6f}")
    print(f"第一层权重梯度范数: {sem_encoder.net[0].weight.grad.norm().item():.6f}")
    print("✓ 梯度反向传播测试通过")

    # 测试 5: NaN 检测
    print("\n测试 5: NaN/Inf 检测")
    try:
        x_nan = torch.randn(16, 640)
        x_nan[0, 0] = float("nan")
        u = sem_encoder(x_nan)
        print("✗ NaN 检测失败：应该抛出异常")
    except RuntimeError as e:
        print(f"✓ NaN 检测成功：{e}")

    try:
        x_inf = torch.randn(16, 640)
        x_inf[0, 0] = float("inf")
        u = sem_encoder(x_inf)
        print("✗ Inf 检测失败：应该抛出异常")
    except RuntimeError as e:
        print(f"✓ Inf 检测成功：{e}")

    # 测试 6: 权重初始化
    print("\n测试 6: 权重初始化检查 (Kaiming)")
    encoder = LatentEncoder(640, 256)
    first_layer_weight = encoder.net[0].weight
    weight_std = first_layer_weight.std().item()
    expected_std = (2.0 / 640) ** 0.5  # Kaiming 推荐值: sqrt(2/fan_in)
    print(f"第一层权重标准差: {weight_std:.6f} (期望约 {expected_std:.3f})")
    assert expected_std * 0.5 < weight_std < expected_std * 1.5, (
        f"权重标准差异常: {weight_std}"
    )
    print("✓ 权重初始化检查通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
