"""
可视化工具模块
"""

import os
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from typing import Optional, List

_cjk_candidates = [
    'Microsoft YaHei',
    'SimHei',
    'Noto Sans CJK SC',
    'Source Han Sans SC',
    'PingFang SC',
    'WenQuanYi Zen Hei',
    'Arial Unicode MS'
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_selected_font = next((n for n in _cjk_candidates if n in _available_fonts), 'DejaVu Sans')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [_selected_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _is_interactive_backend() -> bool:
    try:
        backend = matplotlib.get_backend()
    except Exception:
        return False
    interactive_backends = {"TkAgg", "Qt5Agg", "QtAgg", "WebAgg", "MacOSX", "nbAgg"}
    return backend in interactive_backends


def _safe_filename(title: str) -> str:
    name = "".join(c if c.isalnum() or c in "-_." else "_" for c in title.strip())
    return name or "figure"


def _default_save_path(title: str) -> str:
    out_dir = os.path.join(".", "figures")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{_safe_filename(title)}.png")


def _finalize_figure(fig, title: str, save_path: str | None = None) -> None:
    if save_path is None and not _is_interactive_backend():
        save_path = _default_save_path(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if _is_interactive_backend() and save_path is None:
        plt.show()
    plt.close(fig)


def plot_training_curve(train_losses, train_accuracies, val_accuracies=None, title="Training Curve", val_epochs=None, save_path=None):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表（可选）
        title: 图表标题
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制训练损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建第二个y轴用于准确率
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_accuracies, color=color, label='Train Accuracy')
    
    # 绘制验证准确率（如果提供）
    if val_accuracies is not None and len(val_accuracies) > 0:
        val_x = val_epochs if val_epochs is not None else list(range(10, len(train_losses) + 1, 10))
        ax2.plot(val_x, val_accuracies, color='tab:green', label='Val Accuracy')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title)
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


def plot_epoch_accuracy(epoch_accuracies, title="Epoch Average Accuracy", save_path=None):
    """
    绘制epoch平均准确率曲线
    
    Args:
        epoch_accuracies: epoch平均准确率列表
        title: 图表标题
        save_path: 保存路径（可选）
    """
    epochs = range(1, len(epoch_accuracies) + 1)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_accuracies, 'b-', linewidth=2, marker='o', markersize=6, label='Epoch Accuracy')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title(title)
    plt.legend()
    
    # 设置y轴范围
    plt.ylim(0, 1)
    
    # 添加数值标注（每5个epoch标注一次）
    for i in range(0, len(epoch_accuracies), 5):
        plt.annotate(f'{epoch_accuracies[i]:.3f}', 
                    (epochs[i], epoch_accuracies[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


def plot_epoch_statistics(epoch_means, epoch_stds=None, title="Epoch Statistics", save_path=None):
    """
    绘制epoch统计信息（平均值和标准差）
    
    Args:
        epoch_means: epoch平均准确率列表
        epoch_stds: epoch标准差列表（可选）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    epochs = range(1, len(epoch_means) + 1)
    
    fig = plt.figure(figsize=(12, 6))
    
    # 绘制平均准确率
    plt.plot(epochs, epoch_means, 'b-', linewidth=2, marker='o', markersize=6, label='Mean Accuracy')
    
    # 如果提供了标准差，绘制误差带
    if epoch_stds is not None:
        epoch_means_arr = np.array(epoch_means)
        epoch_stds_arr = np.array(epoch_stds)
        plt.fill_between(epochs, 
                        epoch_means_arr - epoch_stds_arr, 
                        epoch_means_arr + epoch_stds_arr, 
                        alpha=0.2, color='blue', label='±1 Std Dev')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    
    # 设置y轴范围
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    _finalize_figure(fig, title, save_path)


def plot_accuracy_comparison(train_accuracies, val_accuracies=None, test_accuracies=None, 
                           title="Accuracy Comparison", save_path=None, val_epochs=None, test_epochs=None):
    """
    比较训练、验证和测试准确率
    
    Args:
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表（可选）
        test_accuracies: 测试准确率列表（可选）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    epochs = range(1, len(train_accuracies) + 1)
    
    fig = plt.figure(figsize=(12, 6))
    
    # 绘制训练准确率
    plt.plot(epochs, train_accuracies, 'b-', linewidth=2, marker='o', 
             markersize=4, label='Train Accuracy')
    
    # 绘制验证准确率
    if val_accuracies is not None:
        if val_epochs is None:
            if len(val_accuracies) == len(train_accuracies):
                val_epochs = range(1, len(val_accuracies) + 1)
            else:
                val_epochs = list(range(10, len(train_accuracies) + 1, 10))
        plt.plot(val_epochs, val_accuracies, 'g-', linewidth=2, marker='s', 
                 markersize=4, label='Validation Accuracy')
    
    # 绘制测试准确率
    if test_accuracies is not None:
        if test_epochs is None:
            test_epochs = range(1, len(test_accuracies) + 1)
        plt.plot(test_epochs, test_accuracies, 'r-', linewidth=2, marker='^', 
                 markersize=4, label='Test Accuracy')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    
    # 设置y轴范围
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    _finalize_figure(fig, title, save_path)


def plot_accuracy_heatmap(accuracy_matrix, class_names=None, metric_names=None, title="Accuracy Heatmap", save_path=None):
    """
    绘制准确率热力图（用于分析不同类别或条件下的准确率）
    
    Args:
        accuracy_matrix: 准确率矩阵，形状为[n_conditions, n_metrics]
        class_names: 类别名称列表（可选，用作行标签）
        metric_names: 指标名称列表（可选，用作列标签）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    fig = plt.figure(figsize=(12, 8))
    
    # 检测数据类型并设置合适的颜色映射和范围
    matrix = np.array(accuracy_matrix)
    
    # 如果数据包含标准差（通常在第三列），需要特殊处理
    if matrix.shape[1] >= 3:
        # 分别处理准确率列（0,1）和标准差列（2+）
        acc_data = matrix[:, :2]  # 准确率数据
        std_data = matrix[:, 2:]  # 标准差数据
        
        # 标准化处理：将标准差缩放到0-1范围以便可视化
        if std_data.size > 0:
            std_normalized = (std_data - std_data.min()) / (std_data.max() - std_data.min() + 1e-8)
            # 重新组合数据
            display_matrix = np.concatenate([acc_data, std_normalized], axis=1)
        else:
            display_matrix = acc_data
    else:
        display_matrix = matrix
    
    # 创建热力图
    im = plt.imshow(display_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # 设置行标签（条件/类别）
    if class_names is not None:
        plt.yticks(range(len(class_names)), class_names, fontproperties=FontProperties(family=_selected_font))
    
    # 设置列标签（指标）
    if metric_names is not None:
        plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right', fontproperties=FontProperties(family=_selected_font))
    else:
        plt.xlabel('Metrics')
    
    plt.ylabel('Epochs/Conditions')
    plt.title(title)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    if matrix.shape[1] >= 3:
        cbar.set_label('Normalized Values (Accuracy: 0-1, Std: normalized)')
    else:
        cbar.set_label('Accuracy')
    
    # 在每个格子中显示原始数值
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # 显示原始值，不是标准化后的值
            value = matrix[i, j]
            if j < 2:  # 准确率列
                text_str = f'{value:.3f}'
            else:  # 标准差列
                text_str = f'{value:.4f}'
            
            # 根据背景颜色选择文字颜色
            text_color = "white" if display_matrix[i, j] < 0.5 else "black"
            plt.text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=9)
    
    plt.tight_layout()
    
    _finalize_figure(fig, title, save_path)


def plot_val_accuracy_curve(val_accuracies, title="Validation Accuracy Curve", val_epochs=None, save_path=None):
    """
    绘制验证准确率曲线
    
    Args:
        val_accuracies: 验证准确率列表
        title: 图表标题
        val_epochs: epoch列表（可选）
        save_path: 保存路径（可选）
    """
    if isinstance(val_accuracies, torch.Tensor):
        val_accuracies = val_accuracies.detach().cpu().numpy()
    if val_epochs is None:
        val_epochs = range(1, len(val_accuracies) + 1)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, val_accuracies, 'g-', linewidth=2, marker='s', markersize=6, label='Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


def plot_separation_ratio_curve(sep_ratios, sep_epochs=None, title="Separation Ratio Curve", save_path=None):
    """
    绘制分离比曲线
    
    Args:
        sep_ratios: 分离比列表
        sep_epochs: epoch列表（可选）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    if isinstance(sep_ratios, torch.Tensor):
        sep_ratios = sep_ratios.detach().cpu().numpy()
    if sep_epochs is None:
        sep_epochs = list(range(5, 5 * len(sep_ratios) + 1, 5))
    fig = plt.figure(figsize=(10, 6))
    plt.plot(sep_epochs, sep_ratios, 'm-', linewidth=2, marker='d', markersize=6, label='Separation Ratio')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Separation Ratio')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)
