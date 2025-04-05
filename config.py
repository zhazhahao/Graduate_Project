class Config:
    # 数据集配置
    data_dir = r'D:\pycharmprojects\graduate_project\market1501'  # 使用绝对路径
    num_classes = 1501
    input_size = 224
    num_workers = 4
    
    # 训练配置
    batch_size = 8
    epochs = 100  # 增加到100轮（原来50轮+新增50轮）
    lr = 0.001
    
    # 模型配置
    model_type = 'lgca'  # 可选: 'cbam', 'lgca', 'both'
    model_config = {
        'cbam': {
            'enabled': False,  # 是否启用CBAM模型
            'pretrained': '',  # CBAM预训练模型路径
        },
        'lgca': {
            'enabled': True,  # 是否启用LGCA模型
            'pretrained': '',  # 清空预训练路径，因为我们使用resume
        }
    }
    
    # 损失函数配置
    loss_config = {
        'ce_weight': 1.0,      # 交叉熵损失权重
        'triplet_weight': 1.0,  # 三元组损失权重
        'margin': 0.3,         # 三元组损失margin
    }
    
    # 保存配置
    save_dir = r'D:\pycharmprojects\graduate_project\checkpoints'  # 使用绝对路径
    save_freq = 10  # 每多少个epoch保存一次
    
    # 其他配置
    seed = 42
    device = 'cuda'  # 'cuda' or 'cpu'
    resume = r'D:\pycharmprojects\graduate_project\checkpoints\checkpoint_epoch47.pth'  # 设置恢复训练的检查点路径

    def __init__(self):
        # 根据model_type自动设置enabled状态
        if self.model_type == 'cbam':
            self.model_config['cbam']['enabled'] = True
            self.model_config['lgca']['enabled'] = False
        elif self.model_type == 'lgca':
            self.model_config['cbam']['enabled'] = False
            self.model_config['lgca']['enabled'] = True
        elif self.model_type == 'both':
            self.model_config['cbam']['enabled'] = True
            self.model_config['lgca']['enabled'] = True 