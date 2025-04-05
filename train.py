import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import torch.backends.cudnn
from config import Config
from models.resnet_cbam import ResNet50_CBAM
from models.resnet_lgca import ResNet50_LGCA
from utils.data_loader import create_dataloaders
from utils.trainer import ModelTrainer
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualizer import TrainingVisualizer
from utils.losses import CombinedLoss

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # 加载配置
    cfg = Config()
    
    # 设置随机种子
    set_seed(cfg.seed)
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        cfg.data_dir, 
        cfg.batch_size,
        cfg.num_workers
    )
    
    # 创建模型
    models = {}
    optimizers = {}
    schedulers = {}
    
    # 根据配置初始化模型
    if cfg.model_config['cbam']['enabled']:
        models['cbam'] = ResNet50_CBAM(num_classes=cfg.num_classes).to(device)
        optimizers['cbam'] = optim.Adam(models['cbam'].parameters(), lr=cfg.lr)
        schedulers['cbam'] = optim.lr_scheduler.ReduceLROnPlateau(
            optimizers['cbam'], mode='min', factor=0.1, patience=10
        )
        
    if cfg.model_config['lgca']['enabled']:
        models['lgca'] = ResNet50_LGCA(num_classes=cfg.num_classes).to(device)
        optimizers['lgca'] = optim.Adam(models['lgca'].parameters(), lr=cfg.lr)
        schedulers['lgca'] = optim.lr_scheduler.ReduceLROnPlateau(
            optimizers['lgca'], mode='min', factor=0.1, patience=10
        )
    
    # 创建组合损失函数
    criterion = CombinedLoss(
        ce_weight=cfg.loss_config['ce_weight'],
        triplet_weight=cfg.loss_config['triplet_weight'],
        margin=cfg.loss_config['margin']
    ).to(device)
    
    # 创建训练器
    trainer = ModelTrainer(
        models=models,
        criterion=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        device=device,
        model_config=cfg.model_config
    )
    
    # 创建可视化器
    visualizer = TrainingVisualizer(save_dir=os.path.join(cfg.save_dir, 'visualization'))
    
    # 加载检查点（如果存在）
    start_epoch = 0
    best_acc: dict[str, float] = {model_name: 0.0 for model_name in models.keys()}
    
    if cfg.resume:
        start_epoch, best_acc, _ = load_checkpoint(  # type: ignore
            cfg.resume,
            models,
            optimizers,
            schedulers
        )
    
    # 开始训练
    print("\n开始训练...")
    print(f"训练设备: {device}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"总epoch数: {cfg.epochs}")
    print("=" * 50 + "\n")
    
    for epoch in range(start_epoch, cfg.epochs):
        # 训练一个epoch
        train_metrics = trainer.train_epoch(epoch, train_loader)
        
        # 验证
        val_metrics = trainer.validate(epoch, val_loader)
        
        # 更新可视化器
        visualizer.update_metrics(epoch, train_metrics, val_metrics)
        
        # 更新学习率
        for model_name in models.keys():
            schedulers[model_name].step(val_metrics[model_name]['loss'])
        
        # 保存最佳模型
        is_best = False
        for model_name in models.keys():
            if val_metrics[model_name]['acc'] > best_acc[model_name]:
                is_best = True
                best_acc[model_name] = val_metrics[model_name]['acc']
        
        # 保存检查点
        if is_best or epoch % cfg.save_freq == 0:
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
            }
            
            # 添加每个模型的状态
            for model_name in models.keys():
                state.update({
                    f'model_{model_name}_state_dict': models[model_name].state_dict(),
                    f'optimizer_{model_name}_state_dict': optimizers[model_name].state_dict(),
                    f'scheduler_{model_name}_state_dict': schedulers[model_name].state_dict(),
                })
            
            save_checkpoint(
                state=state,
                save_dir=cfg.save_dir,
                is_best=is_best,
                filename=f'checkpoint_epoch{epoch}.pth'
            )
        
        # 每个epoch结束时更新可视化
        if (epoch + 1) % 1 == 0 or epoch == cfg.epochs - 1:  # 每5个epoch或最后一个epoch
            visualizer.plot_training_curves(f'training_curves_epoch_{epoch+1}.png')
            visualizer.plot_loss_components(f'loss_components_epoch_{epoch+1}.png')
    
    # 训练结束，保存最终的可视化结果和指标总结
    print("\n训练完成！保存最终结果...")
    visualizer.plot_training_curves('final_training_curves.png')
    visualizer.plot_loss_components('final_loss_components.png')
    visualizer.save_metrics_summary('final_training_summary.txt')
    print(f"\n所有结果已保存到: {visualizer.save_dir}")

if __name__ == '__main__':
    main() 