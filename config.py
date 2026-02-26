import torch


class Config:
    # 数据配置 - PACS域泛化设置
    dataset = "PACS"
    n_way = 5
    k_shot = 5
    query_per_class = 15
    test_episodes = 2000

    # PACS域配置
    n_domains = 4  # PACS有4个域: photo, art_painting, cartoon, sketch
    source_domains = ["photo", "art_painting", "cartoon"]  # 默认源域
    target_domain = "sketch"  # 默认目标域

    # 训练配置
    num_epochs = 60
    episodes_per_epoch = 150  # 每个 epoch 训练 170 个 episode
    learning_rate = 0.05
    optimizer = "SGD"
    momentum = 0.9
    weight_decay = 5e-4  # 权重衰减系数（L2正则化系数）：在每次参数更新时对权重施加惩罚，防止模型过拟合；5e-4 表示对权重乘以 (1 - 5e-4 * lr) 进行衰减，值越大正则化越强
    nesterov = True  # SGD的Nesterov动量
    use_wandb = False  # 训练阶段开启，测试阶段关闭（测试脚本会显式关闭）

    # 验证配置
    val_episodes = 150  # 每次常规验证运行的episode数
    eval_frequency = 5  # 每n个epoch进行一次常规验证

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True
    amp_dtype = "bf16"

    # 学习率调度配置
    # 线性预热（Linear Warmup）参数
    warmup = True  # 是否启用预热阶段。
    warmup_epochs = 10  # 预热的 epoch 数，必须满足 0 <= warmup_epochs < num_epochs
    warmup_start_lr = (
        1e-3  # 预热起始学习率，必须满足 0 < warmup_start_lr < learning_rate
    )
    scheduler = "MultiStepLR"  # 学习率调度器类型：MultiStepLR 表示“多步长衰减”，会在指定的 epoch 里程碑处将学习率乘以 scheduler_gamma
    scheduler_milestones = [
        40,
        50,
    ]  # 在 40、50 epoch 处进行衰减，使 60 epoch 时学习率足够小
    scheduler_gamma = 0.5  # 每次衰减乘以 0.5，确保 90 epoch 时学习率降到较低值

    # 其他配置
    seed = 321  # 随机种子，确保实验可重复
    log_interval = 10  # 每n个打印一次训练日志

    # 梯度裁剪配置
    grad_clip_norm = 1.0  # 梯度裁剪的最大范数，确保训练稳定性

    # 模型保存配置
    save_best_only = True  # 只保存验证效果最好的模型

    """
    模型配置
    """
    # ExpB1Model模型参数
    metric = "euclidean"  # 原型网络距离度量
    proto_temperature = 5.0  # 原型网络温度参数（用于缩放距离/相似度）
    intrinsic_encoder_drop_rate = 0.2  # 本征编码器ResNet12的Dropout率
    use_blurpool = True  # 是否使用BlurPool进行抗锯齿下采样（默认启用）

    # ST-OLR配置
    use_stolr = True  # 是否使用ST-OLR双分支架构
    sem_olr_rank = 128  # 语义OLR投影器的秩（与ExpB1Model默认值一致）
    sem_olr_mlp_hidden = 512  # 语义OLR投影器的MLP隐藏层维度
    style_olr_rank = 32  # 风格OLR投影器的秩（与ExpB1Model默认值一致）
    style_olr_mlp_hidden = 256  # 风格OLR投影器的MLP隐藏层维度
    token_dim = 128  # Token维度
    n_transformer_layers = 2  # Transformer层数
    n_attention_heads = 4  # 注意力头数
    detach_style_inputs = True  # 是否detach风格分支的输入

    # ST-OLR损失权重
    loss_weight_cls = 1.0  # 分类损失权重
    loss_weight_domain = 0.1  # 域分类损失权重
    loss_weight_hsic = (
        0.1  # HSIC 独立性损失权重 (从0.01提升至0.1，因HSIC原始值通常在1e-3~1e-1)
    )

    # ===== 反事实一致性配置 =====
    loss_weight_cf_sema = 0.5  # 反事实语义不变性损失权重
    loss_weight_sf = 0.2  # 风格跟随损失权重

    # Warmup 调度
    cf_start_epoch = 6  # 反事实损失开始生效的 epoch（1-indexed，前 5 个 epoch 不启用）
    cf_rampup_epochs = 5  # 从 0 线性升到目标权重的 epoch 数

    # 打印设备信息
    @classmethod
    def print_device_info(cls):
        if torch.cuda.is_available():
            print("✅ 成功调用GPU!")
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   当前设备: {cls.device}")
        else:
            print("⚠️  GPU不可用，使用CPU")
            print(f"   当前设备: {cls.device}")

    # 验证损失权重配置
    @classmethod
    def validate_loss_weights(cls):
        """
        验证损失权重配置的有效性

        Raises:
            ValueError: 当权重为负数时
        """
        if cls.loss_weight_cls < 0:
            raise ValueError(f"分类损失权重必须非负，得到: {cls.loss_weight_cls}")
        if cls.loss_weight_domain < 0:
            raise ValueError(f"域分类损失权重必须非负，得到: {cls.loss_weight_domain}")
        if cls.loss_weight_hsic < 0:
            raise ValueError(
                f"HSIC独立性损失权重必须非负，得到: {cls.loss_weight_hsic}"
            )
        if cls.loss_weight_cf_sema < 0:
            raise ValueError(
                f"反事实语义损失权重必须非负，得到: {cls.loss_weight_cf_sema}"
            )
        if cls.loss_weight_sf < 0:
            raise ValueError(f"风格跟随损失权重必须非负，得到: {cls.loss_weight_sf}")

        print(f"✅ 损失权重配置验证通过:")
        print(f"   分类损失权重: {cls.loss_weight_cls}")
        print(f"   域分类损失权重: {cls.loss_weight_domain}")
        print(f"   HSIC独立性损失权重: {cls.loss_weight_hsic}")
        print(f"   反事实语义损失权重: {cls.loss_weight_cf_sema}")
        print(f"   风格跟随损失权重: {cls.loss_weight_sf}")
        print(
            f"   反事实 warmup: epoch {cls.cf_start_epoch} 开始, {cls.cf_rampup_epochs} epochs 线性升温"
        )
