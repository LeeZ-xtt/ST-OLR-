# 标准库导入
import os
import random
import argparse
import time

# 第三方库导入
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 本地模块导入
from config import Config
from utils.scheduler import Scheduler
from utils.index import (
    compute_episode_accuracy,
    compute_confidence_interval,
    compute_epoch_statistics,
    compute_prototype_separation_ratio,
)
from utils.dataloader_improvement import (
    PACSDataset,
    create_cross_domain_episode_loader,
    get_pacs_transform,
)  # 更新为改进版dataloader
from utils.visualization import (
    plot_epoch_accuracy,
    plot_epoch_statistics,
    plot_training_curve,
    plot_accuracy_comparison,
    plot_accuracy_heatmap,
    plot_val_accuracy_curve,
    plot_separation_ratio_curve,
)
from model.module.stolr_losses import total_loss  # 导入总损失函数
from model.module.counterfactual import counterfactual_forward  # 导入反事实前向函数


def setup_environment(seed):
    """
    设置环境种子以确保实验可重复性

    Args:
        seed (int): 随机种子值

    Note:
        - 设置PyTorch、NumPy、Python random的种子
        - 配置CUDNN为确定性模式（可能影响性能但保证可重复性）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设置cudnn确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config):
    """
    创建ExpB1Model单流架构并部署到GPU

    Args:
        config: 配置对象，包含模型超参数

    Returns:
        ExpB1Model: 实验B.1单流模型实例
    """
    from model.exp_b1_model import ExpB1Model

    # 创建ExpB1Model实例并直接部署到GPU
    model = ExpB1Model(
        metric=config.metric,  # 距离度量方式
        proto_temperature=config.proto_temperature,  # 原型网络温度参数
        intrinsic_encoder_drop_rate=config.intrinsic_encoder_drop_rate,
        sem_olr_rank=config.sem_olr_rank,  # 语义OLR投影器的秩
        sem_olr_mlp_hidden=config.sem_olr_mlp_hidden,  # 语义OLR投影器的MLP隐藏层维度
        style_olr_rank=config.style_olr_rank,  # 风格OLR投影器的秩
        style_olr_mlp_hidden=config.style_olr_mlp_hidden,  # 风格OLR投影器的MLP隐藏层维度
        token_dim=config.token_dim,  # Token维度
        n_domains=config.n_domains,  # 域的数量
        detach_style_inputs=config.detach_style_inputs,  # 是否detach风格分支的输入
        n_transformer_layers=config.n_transformer_layers,  # Transformer层数
        n_attention_heads=config.n_attention_heads,  # 注意力头数
    )

    # 部署到设备
    device = Config.device
    model = model.to(device=device)

    # 应用channels_last内存格式优化（可提升20-30%性能）
    channels_last_enabled = False
    if torch.cuda.is_available():
        try:
            model = model.to(memory_format=torch.channels_last)
            channels_last_enabled = True
            print("✅ 模型已转换为channels_last内存格式 (预期性能提升20-30%)")
        except RuntimeError as e:
            print(f"⚠️  channels_last转换失败: {e}")
            print("   继续使用默认contiguous格式")
    else:
        print("⚠️  CPU模式不支持channels_last优化")

    print(f"📋 已创建模型: {model.__class__.__name__}")
    print(f"🚀 模型已部署到: {device}")
    print(f"🔧 模型配置: metric={config.metric}")

    return model, channels_last_enabled


def run_episode(
    model,
    support_images,
    support_labels,
    query_images,
    n_way,
    support_domains,
    query_domains,
):
    """
    运行单个episode - ExpB1Model单流架构

    Args:
        model: ExpB1Model实例
        support_images: 支持集图像 [N_s, C, H, W]
        support_labels: 支持集标签 [N_s]
        query_images: 查询集图像 [N_q, C, H, W]
        n_way: 类别数
        support_domains: 支持集域标签 [N_s]
        query_domains: 查询集域标签 [N_q]

    Returns:
        dict: 包含以下键的字典
            - logits: 原型分类 logits [N_q, n_way]
            - prototypes: 计算出的原型 [n_way, 640]
            - z_c: 语义嵌入 [B_total, 640]
            - z_s: 风格嵌入 [B_total, 640]
            - u_c: 语义潜码 [B_total, 256]
            - u_s: 风格潜码 [B_total, 64]
            - domain_logits: 域分类 logits [B_total, 4]
            - support_domains: 支持集域标签 [N_s]
            - query_domains: 查询集域标签 [N_q]

    Note:
        联合 Stiefel + HSIC 架构返回完整的 outputs 字典
    """
    return model(
        support_images,
        support_labels,
        query_images,
        n_way,
        support_domains,
        query_domains,
    )


# 使用 utils.index 中的 compute_epoch_statistics，移除本地重复实现


def evaluate_model(model, dataset, config, num_test_episodes=100):
    """
    评估模型性能
    """
    # 设置为评估模式
    model.set_mode("eval")

    accuracies = []

    # 创建测试episode加载器 - 使用跨域采样
    # 注意：这里我们使用 args.test_source_domains 和 args.test_query_domains
    # 但 evaluate_model 函数签名没有这些参数，我们需要从 config 获取
    episode_loader = create_cross_domain_episode_loader(
        dataset,
        config.n_way,
        config.k_shot,
        config.query_per_class,
        num_test_episodes,
        support_domain_pool=config.test_source_domains,
        query_domain_pool=config.test_query_domains,
    )

    use_amp = getattr(Config, "use_amp", False) and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16
        if getattr(Config, "amp_dtype", "bf16").lower() == "bf16"
        else torch.float16
    )
    with torch.no_grad():
        for episode_idx, episode_data in enumerate(episode_loader):
            if len(episode_data) == 6:
                support_images, support_labels, query_images, query_labels, _, _ = (
                    episode_data
                )
            else:
                support_images, support_labels, query_images, query_labels = (
                    episode_data[:4]
                )
            # 移动到设备
            support_images = support_images.to(config.device)
            support_labels = support_labels.to(config.device)
            query_images = query_images.to(config.device)
            query_labels = query_labels.to(config.device)

            # 准备域标签（评估时使用虚拟域标签）
            support_domains_eval = torch.zeros(
                support_images.shape[0], dtype=torch.long, device=config.device
            )
            query_domains_eval = torch.zeros(
                query_images.shape[0], dtype=torch.long, device=config.device
            )

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = run_episode(
                        model,
                        support_images,
                        support_labels,
                        query_images,
                        config.n_way,
                        support_domains_eval,
                        query_domains_eval,
                    )
                    logits = outputs["logits"]
            else:
                outputs = run_episode(
                    model,
                    support_images,
                    support_labels,
                    query_images,
                    config.n_way,
                    support_domains_eval,
                    query_domains_eval,
                )
                logits = outputs["logits"]

            # 计算准确率
            acc = compute_episode_accuracy(logits, query_labels)
            accuracies.append(acc)

            if (episode_idx + 1) % 10 == 0:
                print(f"  Evaluated {episode_idx + 1}/{num_test_episodes} episodes")

    # 计算平均准确率和置信区间
    mean_acc = np.mean(accuracies)
    lower_bound, upper_bound = compute_confidence_interval(accuracies)

    # 重置为训练模式
    model.set_mode("train")

    return mean_acc, lower_bound, upper_bound


def main():
    """
    主训练函数
    """
    # 解析命令行参数 - 更新为PACS数据集路径
    parser = argparse.ArgumentParser(
        description="Train ExpB1Model - Dual Stream for Domain Generalization (Improved Sampling)"
    )
    parser.add_argument(
        "--pacs_root",
        type=str,
        required=True,
        help="Path to PACS dataset root directory",
    )

    # 训练阶段域配置
    parser.add_argument(
        "--train_source_domains",
        nargs="+",
        default=["photo", "art_painting", "cartoon"],
        help="Source domains for training support set",
    )
    parser.add_argument(
        "--train_query_domains",
        nargs="+",
        default=["photo", "art_painting", "cartoon"],
        help="Domains for training query set (will be sampled consistently per episode)",
    )

    # 验证阶段域配置
    parser.add_argument(
        "--test_source_domains",
        nargs="+",
        default=["photo", "art_painting", "cartoon"],
        help="Source domains for validation support set",
    )
    parser.add_argument(
        "--test_query_domains",
        nargs="+",
        default=["sketch"],
        help="Target domain for validation query set",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=Config.num_epochs,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--episodes_per_epoch",
        type=int,
        default=Config.episodes_per_epoch,
        help="Number of episodes per epoch",
    )

    args = parser.parse_args()

    # 更新配置
    Config.num_epochs = args.num_epochs
    Config.episodes_per_epoch = args.episodes_per_epoch

    # 将参数注入 Config 以便在 evaluate_model 中使用
    Config.train_source_domains = args.train_source_domains
    Config.train_query_domains = args.train_query_domains
    Config.test_source_domains = args.test_source_domains
    Config.test_query_domains = args.test_query_domains

    # 设置环境
    setup_environment(Config.seed)

    # 打印设备信息
    Config.print_device_info()

    # 验证损失权重配置
    Config.validate_loss_weights()

    # 创建模型（已包含GPU部署）- 使用完整配置
    model, channels_last_enabled = create_model(Config)

    # 获取模型参数组（Patch 3: 支持参数分组）
    param_groups = model.get_parameters()

    # 优化器配置 - 使用配置参数
    optimizer = torch.optim.SGD(
        param_groups,
        lr=Config.learning_rate,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay,
        nesterov=Config.nesterov,
    )

    # 提取所有参数用于梯度裁剪（从参数组中展开）
    all_params = []
    for group in param_groups:
        all_params.extend(group["params"])

    # 组合调度器：线性预热 + MultiStepLR
    scheduler = Scheduler(optimizer)

    # 损失函数配置 - 使用 stolr_losses.total_loss_val 统一计算
    criterion = nn.CrossEntropyLoss()

    use_amp = getattr(Config, "use_amp", False) and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16
        if getattr(Config, "amp_dtype", "bf16").lower() == "bf16"
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    print(f"🔧 训练配置:")
    print(f"   学习率: {Config.learning_rate} (预热: {Config.warmup_epochs} epochs)")
    print(f"   梯度裁剪: {Config.grad_clip_norm}")
    print(f"   验证频率: 每 {Config.eval_frequency} epochs")
    print(f"   AMP: {use_amp} (dtype={getattr(Config, 'amp_dtype', 'bf16')})")

    # 数据预处理 - 使用PACS专用transform
    train_transform = get_pacs_transform(image_size=84, split="train")
    eval_transform = get_pacs_transform(image_size=84, split="test")

    # 加载PACS数据集 - 域泛化设置
    # 训练集需要包含所有训练阶段用到的域
    train_domains = list(set(args.train_source_domains + args.train_query_domains))
    train_dataset = PACSDataset(
        root_dir=args.pacs_root,
        target_domains=train_domains,
        split="train",
        transform=train_transform,
    )

    # 验证集需要包含所有验证阶段用到的域
    val_domains = list(set(args.test_source_domains + args.test_query_domains))
    val_dataset = PACSDataset(
        root_dir=args.pacs_root,
        target_domains=val_domains,
        split="test",  # PACS没有独立的val集，使用test作为验证
        transform=eval_transform,
    )

    # T-SNE 专用固定数据集 (使用eval_transform避免随机增强，确保可视化一致性)
    # 注意：为了对齐域泛化评估（source→target），需要包含test_source_domains和test_query_domains
    tsne_domains = list(set(args.test_source_domains + args.test_query_domains))
    tsne_dataset = PACSDataset(
        root_dir=args.pacs_root,
        target_domains=tsne_domains,  # 使用验证阶段的域（source + target）
        split="train",
        transform=eval_transform,  # 关键：使用评估时的变换（无随机增强）
    )

    print(f"📊 数据集配置 (Improved):")
    print(f"   训练支持域: {args.train_source_domains}")
    print(f"   训练查询域: {args.train_query_domains}")
    print(f"   验证支持域: {args.test_source_domains}")
    print(f"   验证查询域: {args.test_query_domains}")
    print(f"   训练样本: {len(train_dataset)} 张图像 (涵盖 {train_domains})")
    print(f"   验证样本: {len(val_dataset)} 张图像 (涵盖 {val_domains})")

    # 训练历史记录
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    epoch_stds = []  # 用于记录每个epoch的标准差

    # 热力图数据收集：不同阶段的准确率矩阵
    heatmap_data = []
    sep_ratio_curve = []
    sep_ratio_epochs = []
    best_val_acc = 0.0

    # ===== 权重更新量监控：保存第一层卷积的初始权重 =====
    first_conv_layer = model.backbone.stem_conv[0]  # stem_conv 的第一个 Conv2d
    initial_weights = first_conv_layer.weight.data.clone().detach()
    print(f"📊 权重监控已启用: 跟踪 backbone.stem_conv[0] (shape={initial_weights.shape})")

    print("Starting training...")


    # HSIC 调试配置
    HSIC_DEBUG_EPISODES = 10  # 前 N 个 episode 打印 HSIC 调试信息
    hsic_debug_counter = 0  # 全局 episode 计数器
    hsic_values_log = []  # 记录 HSIC 值用于分析

    # 检查eval_frequency参数的有效性
    if not isinstance(Config.eval_frequency, int) or Config.eval_frequency < 1:
        raise ValueError("eval_frequency must be a positive integer")

    # 记录总训练开始时间
    total_start_time = time.time()
    epoch_times = []  # 记录每个epoch的时间

    # 训练循环
    for epoch in tqdm(range(Config.num_epochs), desc="Training Progress", unit="epoch"):
        # 记录epoch开始时间
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        # 在每个 epoch 开始时更新学习率（满足预热阶段线性增长与主阶段里程碑下降的要求）
        scheduler.step()
        print(f"  Current LR: {scheduler.get_lr():.6f}")

        # 创建训练episode加载器 - 使用改进的跨域采样策略
        episode_loader = create_cross_domain_episode_loader(
            train_dataset,
            Config.n_way,
            Config.k_shot,
            Config.query_per_class,
            Config.episodes_per_epoch,
            support_domain_pool=args.train_source_domains,
            query_domain_pool=args.train_query_domains,
        )

        epoch_losses = []
        epoch_accuracies = []
        episode_times = []

        # 分别记录各项损失
        epoch_cls_losses = []
        epoch_domain_losses = []
        epoch_hsic_losses = []
        epoch_cf_sema_losses = []
        epoch_sf_losses = []

        layer_weights = None
        epoch_sep_ratios = [] if (epoch + 1) % 5 == 0 else None

        # 遍历所有episodes
        episode_loader_with_progress = tqdm(
            episode_loader,
            total=Config.episodes_per_epoch,
            desc=f"Epoch {epoch + 1} Episodes",
            leave=False,
            unit="episode",
        )

        for episode_idx, episode_data in enumerate(episode_loader_with_progress):
            # 处理episode数据（包含域标签）
            if len(episode_data) == 6:
                (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    support_domains,
                    query_domains,
                ) = episode_data
            else:
                # 兼容旧版本数据加载器（无域标签）
                support_images, support_labels, query_images, query_labels = (
                    episode_data[:4]
                )
                # 使用虚拟域标签
                support_domains = torch.zeros(support_images.shape[0], dtype=torch.long)
                query_domains = torch.zeros(query_images.shape[0], dtype=torch.long)

            # 记录episode开始时间
            episode_start_time = time.time()

            # 移动到设备并应用channels_last格式（若已启用）
            if channels_last_enabled:
                support_images = support_images.to(
                    Config.device, memory_format=torch.channels_last
                )
                query_images = query_images.to(
                    Config.device, memory_format=torch.channels_last
                )
            else:
                support_images = support_images.to(Config.device)
                query_images = query_images.to(Config.device)

            support_labels = support_labels.to(Config.device)
            query_labels = query_labels.to(Config.device)
            support_domains = support_domains.to(Config.device)  # 移动域标签到设备
            query_domains = query_domains.to(Config.device)  # 移动域标签到设备

            if support_images.dim() != 4 or query_images.dim() != 4:
                raise ValueError(
                    f"Expected 4D images, got support {support_images.dim()}D, query {query_images.dim()}D"
                )

            # 前向与损失计算（AMP）
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = run_episode(
                        model,
                        support_images,
                        support_labels,
                        query_images,
                        Config.n_way,
                        support_domains,
                        query_domains,
                    )
                    logits = outputs["logits"]  # [N_q, n_way]

                    # 将域标签添加到 outputs 以便 total_loss 使用
                    outputs["support_domains"] = support_domains
                    outputs["query_domains"] = query_domains

                    # ===== 反事实前向（训练期） =====
                    cf_outputs = None
                    if model.training:
                        B_total = outputs["f1"].shape[0]
                        perm = torch.randperm(B_total, device=Config.device)
                        cf_outputs = counterfactual_forward(
                            model,
                            f1=outputs["f1"],
                            f2=outputs["f2"],
                            f3=outputs["f3"],
                            perm=perm,
                        )

                    # 使用 stolr_losses.total_loss 统一计算所有损失（简化接口）
                    losses = total_loss(
                        outputs,
                        query_labels,
                        cf_outputs=cf_outputs,
                        current_epoch=epoch + 1,  # 转换为 1-indexed
                    )
                    cls_loss = losses["cls"]
                    domain_loss = losses["domain"]
                    loss_hsic = losses["hsic"]
                    loss_cf_sema = losses["cf_sema"]
                    loss_sf = losses["sf"]
                    total_loss_val = losses["total"]
            else:
                outputs = run_episode(
                    model,
                    support_images,
                    support_labels,
                    query_images,
                    Config.n_way,
                    support_domains,
                    query_domains,
                )
                logits = outputs["logits"]  # [N_q, n_way]

                # 将域标签添加到 outputs 以便 total_loss 使用
                outputs["support_domains"] = support_domains
                outputs["query_domains"] = query_domains

                # ===== 反事实前向（训练期） =====
                cf_outputs = None
                if model.training:
                    B_total = outputs["f1"].shape[0]
                    perm = torch.randperm(B_total, device=Config.device)
                    cf_outputs = counterfactual_forward(
                        model,
                        f1=outputs["f1"],
                        f2=outputs["f2"],
                        f3=outputs["f3"],
                        perm=perm,
                    )

                # 使用 stolr_losses.total_loss 统一计算所有损失（简化接口）
                losses = total_loss(
                    outputs,
                    query_labels,
                    cf_outputs=cf_outputs,
                    current_epoch=epoch + 1,  # 转换为 1-indexed
                )
                cls_loss = losses["cls"]
                domain_loss = losses["domain"]
                loss_hsic = losses["hsic"]
                loss_cf_sema = losses["cf_sema"]
                loss_sf = losses["sf"]
                total_loss_val = losses["total"]

            # HSIC 调试日志（仅在前 N 个 episode 打印）
            if hsic_debug_counter < HSIC_DEBUG_EPISODES:
                hsic_raw = loss_hsic.item()
                hsic_weighted = Config.loss_weight_hsic * hsic_raw
                hsic_values_log.append(hsic_raw)

                # 计算归一化因子 (B-1)^2
                B_total = outputs["u_c"].shape[0]
                normalization_factor = (B_total - 1) ** 2 if B_total >= 2 else 1

                print(
                    f"\n=== HSIC 调试 (Epoch {epoch + 1}, Episode {episode_idx + 1}) ==="
                )
                print(f"  Batch Size B: {B_total}")
                print(f"  归一化因子 (B-1)^2: {normalization_factor}")
                print(f"  nHSIC 原始值: {hsic_raw:.4f} (值域 [0,1]，0=独立)")
                print(
                    f"  加权后 HSIC 损失 (权重={Config.loss_weight_hsic}): {hsic_weighted:.4e}"
                )

                # nHSIC 值域为 [0, 1]，需要使用匹配的阈值
                if hsic_raw < 0.01:
                    magnitude_status = "✅ 非常好(接近独立)"
                    suggestion = "u_c 与 u_s 已接近独立，HSIC 约束生效"
                elif hsic_raw < 0.3:
                    magnitude_status = "✅ 良好 (弱依赖)"
                    suggestion = "当前权重配置合理"
                elif hsic_raw < 0.7:
                    magnitude_status = "✅ 正常 (中等依赖，训练初期预期行为)"
                    suggestion = f"当前 loss_weight_hsic={Config.loss_weight_hsic} 配置合理，nHSIC 应随训练逐渐下降"
                else:
                    magnitude_status = "⚠️  较高依赖 (> 0.7)"
                    suggestion = f"建议: 可适当增大 loss_weight_hsic 至 {Config.loss_weight_hsic * 2:.4f}"

                print(f"  量级分析: {magnitude_status}")
                print(f"  {suggestion}")

                hsic_debug_counter += 1

                # 在调试阶段结束时给出汇总
                if hsic_debug_counter == HSIC_DEBUG_EPISODES:
                    avg_hsic = np.mean(hsic_values_log)
                    std_hsic = np.std(hsic_values_log)
                    print(f"\n=== HSIC 调试汇总 ({HSIC_DEBUG_EPISODES} episodes) ===")
                    print(f"  nHSIC 均值: {avg_hsic:.4f} (值域 [0,1]，0=独立)")
                    print(f"  nHSIC 标准差: {std_hsic:.4f}")
                    print(
                        f"  nHSIC 范围: [{min(hsic_values_log):.4f}, {max(hsic_values_log):.4f}]"
                    )

                    weighted_avg = Config.loss_weight_hsic * avg_hsic
                    print(f"  加权 HSIC 损失: {weighted_avg:.4e}")

                    if avg_hsic < 0.05:
                        print(f"  状态: 特征已接近独立，HSIC 约束正在生效")
                    elif avg_hsic < 0.5:
                        print(f"  状态: 中等依赖，训练初期正常")
                    else:
                        print(f"  状态: 较高依赖（预期随训练下降）")

                    print(
                        f"  当前 loss_weight_hsic={Config.loss_weight_hsic} 配置合理，无需调整"
                    )

            # 增强的NaN检测：检查所有损失组件并提供详细诊断信息
            if torch.isnan(cls_loss):
                print(
                    f"\n❌ NaN损失检测到在Epoch {epoch + 1}, Episode {episode_idx + 1}:"
                )
                print(f"   分类损失(cls_loss): {cls_loss.item()}")
                print(f"   当前学习率: {scheduler.get_lr():.6f}")
                raise ValueError("分类损失包含NaN，训练终止")

            if torch.isnan(domain_loss):
                print(
                    f"\n❌ NaN损失检测到在Epoch {epoch + 1}, Episode {episode_idx + 1}:"
                )
                print(f"   域分类损失(domain_loss): {domain_loss.item()}")
                print(f"   当前学习率: {scheduler.get_lr():.6f}")
                raise ValueError("域分类损失包含NaN，训练终止")

            if torch.isnan(loss_hsic):
                print(
                    f"\n❌ NaN损失检测到在Epoch {epoch + 1}, Episode {episode_idx + 1}:"
                )
                print(f"   HSIC独立性损失(loss_hsic): {loss_hsic.item()}")
                print(f"   当前学习率: {scheduler.get_lr():.6f}")
                raise ValueError("HSIC独立性损失包含NaN，训练终止")

            # 反事实损失 NaN 检测（自动判断是否激活）
            if epoch + 1 >= Config.cf_start_epoch:
                if torch.isnan(loss_cf_sema):
                    print(
                        f"\n❌ NaN损失检测到在Epoch {epoch + 1}, Episode {episode_idx + 1}:"
                    )
                    print(f"   反事实语义损失(loss_cf_sema): {loss_cf_sema.item()}")
                    raise ValueError("反事实语义损失包含NaN，训练终止")

                if torch.isnan(loss_sf):
                    print(
                        f"\n❌ NaN损失检测到在Epoch {epoch + 1}, Episode {episode_idx + 1}:"
                    )
                    print(f"   风格跟随损失(loss_sf): {loss_sf.item()}")
                    raise ValueError("风格跟随损失包含NaN，训练终止")

            if torch.isnan(total_loss_val) or torch.isinf(total_loss_val):
                print(f"\n❌ 总损失异常在Epoch {epoch + 1}, Episode {episode_idx + 1}:")
                print(f"   总损失(total_loss_val): {total_loss_val.item()}")
                print(
                    f"   分类损失: {cls_loss.item()}, 域损失: {domain_loss.item()}, HSIC损失: {loss_hsic.item()}"
                )
                raise ValueError(
                    f"总损失为NaN/Inf (total_loss_val={total_loss_val.item()})"
                )

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total_loss_val).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    all_params, max_norm=Config.grad_clip_norm
                )

                # 梯度监控：检测梯度爆炸
                total_grad_norm = 0.0
                for p in all_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm**0.5

                # 警告：梯度异常大（超过10倍阈值）
                if total_grad_norm > Config.grad_clip_norm * 10:
                    print(
                        f"\n⚠️  梯度异常大警告 - Epoch {epoch + 1}, Episode {episode_idx + 1}:"
                    )
                    print(
                        f"   总梯度范数: {total_grad_norm:.2f} (阈值: {Config.grad_clip_norm})"
                    )
                    print(f"   当前学习率: {scheduler.get_lr():.6f}")

                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(
                    all_params, max_norm=Config.grad_clip_norm
                )

                # 梯度监控：检测梯度爆炸（非AMP路径）
                total_grad_norm = 0.0
                for p in all_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm**0.5

                if total_grad_norm > Config.grad_clip_norm * 10:
                    print(
                        f"\n⚠️  梯度异常大警告 - Epoch {epoch + 1}, Episode {episode_idx + 1}:"
                    )
                    print(
                        f"   总梯度范数: {total_grad_norm:.2f} (阈值: {Config.grad_clip_norm})"
                    )
                    print(f"   当前学习率: {scheduler.get_lr():.6f}")

                optimizer.step()

            # 计算准确率
            acc = compute_episode_accuracy(logits, query_labels)
            epoch_losses.append(total_loss_val.item())
            epoch_accuracies.append(acc)

            # 分别记录各项损失
            epoch_cls_losses.append(cls_loss.item())
            epoch_domain_losses.append(domain_loss.item())
            epoch_hsic_losses.append(loss_hsic.item())
            epoch_cf_sema_losses.append(loss_cf_sema.item())
            epoch_sf_losses.append(loss_sf.item())

            # 记录episode时间
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)

            if epoch_sep_ratios is not None:
                # 直接从outputs中提取support特征，避免重复计算
                N_s = support_images.shape[0]
                support_features = outputs["z_c"][:N_s]  # [N_s, 640] 语义嵌入
                prototypes = outputs["prototypes"]  # [n_way, 640]
                sep_metrics = compute_prototype_separation_ratio(
                    support_features, support_labels, prototypes
                )
                if "separation_ratio" in sep_metrics:
                    r = sep_metrics["separation_ratio"]
                    epoch_sep_ratios.append(float(r))

            # 打印进度 - 包含所有损失信息
            if (episode_idx + 1) % Config.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-Config.log_interval :])
                avg_acc = np.mean(epoch_accuracies[-Config.log_interval :])
                avg_time = np.mean(episode_times[-Config.log_interval :])

                # 计算各项损失的平均值
                avg_cls_loss = np.mean(epoch_cls_losses[-Config.log_interval :])
                avg_domain_loss = np.mean(epoch_domain_losses[-Config.log_interval :])
                avg_hsic_loss = np.mean(epoch_hsic_losses[-Config.log_interval :])

                print(
                    f"  Episode {episode_idx + 1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={avg_time:.2f}s"
                )
                print(
                    f"    └─ 分类损失={avg_cls_loss:.4f}, 域损失={avg_domain_loss:.4f}, HSIC损失={avg_hsic_loss:.6f}"
                )

                # 反事实损失日志（自动判断是否激活）
                avg_cf_sema = np.mean(epoch_cf_sema_losses[-Config.log_interval :])
                avg_sf = np.mean(epoch_sf_losses[-Config.log_interval :])
                if avg_cf_sema > 1e-6 or avg_sf > 1e-6:  # 如果损失非零，说明已激活
                    # 计算当前 warmup 进度
                    if epoch + 1 >= Config.cf_start_epoch:
                        epochs_since_start = (epoch + 1) - Config.cf_start_epoch
                        cf_weight = min(1.0, epochs_since_start / Config.cf_rampup_epochs)
                    else:
                        cf_weight = 0.0
                    print(
                        f"    └─ 反事实语义={avg_cf_sema:.4f}, 风格跟随={avg_sf:.4f}, cf_weight={cf_weight:.2f}"
                    )

        # 计算epoch统计信息
        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_acc = np.mean(epoch_accuracies)
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(avg_epoch_acc)

        # 计算各项损失的epoch平均值
        avg_epoch_cls_loss = np.mean(epoch_cls_losses)
        avg_epoch_domain_loss = np.mean(epoch_domain_losses)
        avg_epoch_hsic_loss = np.mean(epoch_hsic_losses)
        avg_epoch_cf_sema_loss = np.mean(epoch_cf_sema_losses)
        avg_epoch_sf_loss = np.mean(epoch_sf_losses)

        # 计算并记录epoch统计信息（均值、标准差、标准误差）
        epoch_mean, epoch_std, epoch_se = compute_epoch_statistics(epoch_accuracies)
        epoch_stds.append(epoch_std)

        # 记录epoch时间统计
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_episode_time = np.mean(episode_times)
        total_episode_time = np.sum(episode_times)

        print(
            f"  Epoch Summary: Loss={avg_epoch_loss:.4f}, Acc={avg_epoch_acc:.4f}, Std={epoch_std:.4f}"
        )
        print(
            f"    └─ 分类损失={avg_epoch_cls_loss:.4f}, 域损失={avg_epoch_domain_loss:.4f}, HSIC损失={avg_epoch_hsic_loss:.6f}"
        )
        # 反事实损失日志（自动判断是否激活）
        if avg_epoch_cf_sema_loss > 1e-6 or avg_epoch_sf_loss > 1e-6:
            print(
                f"    └─ 反事实语义={avg_epoch_cf_sema_loss:.4f}, 风格跟随={avg_epoch_sf_loss:.4f}"
            )
        print(
            f"  Time Summary: Epoch={epoch_time:.2f}s, Avg Episode={avg_episode_time:.2f}s"
        )

        # ===== 权重更新量监控：计算并打印第一层卷积权重的更新量范数 =====
        current_weights = first_conv_layer.weight.data  # [out_ch, in_ch, 3, 3]
        weight_update = current_weights - initial_weights  # 计算权重变化量
        update_norm = torch.norm(weight_update).item()  # L2范数
        update_norm_relative = update_norm / (torch.norm(initial_weights).item() + 1e-8)  # 相对更新量
        print(f"  📊 Backbone第一层卷积权重更新量:")
        print(f"    └─ 绝对范数: {update_norm:.6f}")
        print(f"    └─ 相对范数: {update_norm_relative:.6f} (更新量/初始权重范数，相对初始权重更新了多少)")
        print(f"    └─ 当前权重范数: {torch.norm(current_weights).item():.6f}")

        # 每 5 个 epoch 记录一次分离比和T-SNE可视化
        if (epoch + 1) % 5 == 0:
            # 若本 epoch 计算了分离比，则累加到曲线数据并打印均值
            if epoch_sep_ratios is not None and len(epoch_sep_ratios) > 0:
                avg_sep_ratio = float(np.mean(epoch_sep_ratios))
                sep_ratio_curve.append(avg_sep_ratio)
                sep_ratio_epochs.append(epoch + 1)
                print(f"  Separation Ratio (avg over epoch): {avg_sep_ratio:.4f}")

            # ===== A_s 梯度诊断 =====
            # 自动判断反事实损失是否激活（通过检查当前 epoch）
            if epoch + 1 >= Config.cf_start_epoch:
                orig_weight = model.joint_basis._basis.parametrizations.weight.original
                if orig_weight.grad is not None:
                    grad_norm = orig_weight.grad.norm().item()
                    print(f"  📊 Q (Stiefel basis) gradient norm: {grad_norm:.6f}")
                else:
                    print(f"  ⚠️  Q gradient is None (反事实损失可能未正确回流)")

        # 根据配置参数评估验证集性能
        if (epoch + 1) % Config.eval_frequency == 0:
            print("  Evaluating on validation set...")
            val_acc, val_lower, val_upper = evaluate_model(
                model, val_dataset, Config, num_test_episodes=Config.val_episodes
            )
            val_accuracies.append(val_acc)
            print(
                f"  Validation Accuracy: {val_acc:.4f} ({val_lower:.4f} ~ {val_upper:.4f})"
            )

            # 收集热力图数据：[epoch, train_acc, val_acc, std]
            heatmap_data.append([epoch + 1, avg_epoch_acc, val_acc, epoch_std])

            # 保存最佳模型
            is_best_model = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best_model = True
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "model_name": model.__class__.__name__,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_acc": best_val_acc,
                    },
                    "best_model.pth",
                )
                print(
                    f"  Saved new best model with validation accuracy: {best_val_acc:.4f}"
                )

            # 每 10 个 epoch 绘制一次验证准确率曲线（若已收集到验证结果）
            if (epoch + 1) % 10 == 0:
                # 生成已评估的 epoch 列表
                current_val_epochs = [
                    i * Config.eval_frequency for i in range(1, len(val_accuracies) + 1)
                ]
                plot_val_accuracy_curve(
                    val_accuracies,
                    title=f"Validation Accuracy Curve (up to Epoch {epoch + 1})",
                    val_epochs=current_val_epochs,
                    save_path=f"figures/val_accuracy_curve_epoch_{epoch + 1}.png",
                )
                # 若已计算分离比，则同步绘制分离比变化曲线
                if len(sep_ratio_curve) > 0:
                    plot_separation_ratio_curve(
                        sep_ratios=sep_ratio_curve,
                        sep_epochs=sep_ratio_epochs,
                        title=f"Separation Ratio Curve (up to Epoch {epoch + 1})",
                        save_path=f"figures/separation_ratio_curve_epoch_{epoch + 1}.png",
                    )

    # 绘制改进的训练曲线 - 使用专业的准确率比较图
    # 根据实际验证次数计算val_epochs
    val_epochs = list(
        range(
            Config.eval_frequency,
            len(val_accuracies) * Config.eval_frequency + 1,
            Config.eval_frequency,
        )
    )

    # 绘制传统的训练曲线（包含损失）
    plot_training_curve(
        train_losses,
        train_accuracies,
        val_accuracies,
        title="Complete Training Curve",
        val_epochs=val_epochs,
        save_path="figures/complete_training_curve.png",
    )

    # 绘制训练与验证准确率对比图，直观展示模型在训练集与验证集上的性能差异
    plot_accuracy_comparison(
        train_accuracies,
        val_accuracies,
        title="Training vs Validation Accuracy Comparison",
        val_epochs=val_epochs,
        save_path="figures/accuracy_comparison.png",
    )
    # 绘制训练阶段最终统计图，展示各 epoch 的平均准确率与标准差
    plot_epoch_statistics(
        train_accuracies,
        epoch_stds,
        title="Training Statistics (Final)",
        save_path="figures/epoch_stats_final.png",
    )
    # 单独绘制训练阶段各 epoch 平均准确率曲线，便于观察整体趋势
    plot_epoch_accuracy(
        train_accuracies,
        title="Epoch Average Accuracy (Final)",
        save_path="figures/epoch_accuracy_final.png",
    )

    # 绘制准确率热力图 - 行为不同epoch，列为指标
    if len(heatmap_data) > 0:
        # 转换为numpy数组：形状 [n_epochs, 3] -> (train_acc, val_acc, std)
        heatmap_matrix = np.array(heatmap_data)[:, 1:]  # 不转置，按 epoch 为行

        # 创建标签
        epoch_labels = [f"Epoch {int(data[0])}" for data in heatmap_data]
        metric_labels = ["训练准确率", "验证准确率", "标准差"]

        plot_accuracy_heatmap(
            heatmap_matrix,
            class_names=epoch_labels,
            metric_names=metric_labels,
            title="Training Progress Heatmap - Accuracy & Statistics",
            save_path="figures/training_progress_heatmap.png",
        )

        print(
            f"  Generated training progress heatmap with {len(heatmap_data)} validation points"
        )

    # 计算并输出总训练时间统计
    total_training_time = time.time() - total_start_time
    avg_epoch_time = np.mean(epoch_times)
    total_episodes = Config.num_epochs * Config.episodes_per_epoch
    avg_episode_time_overall = total_training_time / total_episodes

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED - TIME STATISTICS")
    print("=" * 60)
    print(
        f"Total Training Time: {total_training_time:.2f}s ({total_training_time / 3600:.2f}h)"
    )
    print(f"Number of Epochs: {Config.num_epochs}")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Time per Episode: {avg_episode_time_overall:.2f}s")
    print(f"Episodes per Epoch: {Config.episodes_per_epoch}")
    print("=" * 60)
    print("All done!")


if __name__ == "__main__":
    main()
