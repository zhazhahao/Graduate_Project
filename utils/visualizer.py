import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，这是一个非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

class TrainingVisualizer:
    def __init__(self, save_dir='visualization'):
        """初始化可视化器
        
        Args:
            save_dir: 基础保存目录
        """
        # 创建带时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f'run_{timestamp}')
        self.metrics_history = {
            'train': {'cbam': {'loss': [], 'acc': [], 'ce_loss': [], 'triplet_loss': []},
                     'lgca': {'loss': [], 'acc': [], 'ce_loss': [], 'triplet_loss': []}},
            'val': {'cbam': {'loss': [], 'acc': [], 'ce_loss': [], 'triplet_loss': []},
                   'lgca': {'loss': [], 'acc': [], 'ce_loss': [], 'triplet_loss': []}}
        }
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('default')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # 创建训练日志文件
        self.log_file = os.path.join(self.save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {timestamp}\n")
            f.write("=" * 50 + "\n\n")

    def update_metrics(self, epoch, train_metrics, val_metrics):
        """更新指标历史记录并保存到日志"""
        for model_name in train_metrics.keys():
            # 更新训练指标
            self.metrics_history['train'][model_name]['loss'].append(train_metrics[model_name]['loss'])
            self.metrics_history['train'][model_name]['acc'].append(train_metrics[model_name]['acc'])
            self.metrics_history['train'][model_name]['ce_loss'].append(train_metrics[model_name]['ce_loss'])
            self.metrics_history['train'][model_name]['triplet_loss'].append(train_metrics[model_name]['triplet_loss'])
            
            # 更新验证指标
            self.metrics_history['val'][model_name]['loss'].append(val_metrics[model_name]['loss'])
            self.metrics_history['val'][model_name]['acc'].append(val_metrics[model_name]['acc'])
            self.metrics_history['val'][model_name]['ce_loss'].append(val_metrics[model_name]['ce_loss'])
            self.metrics_history['val'][model_name]['triplet_loss'].append(val_metrics[model_name]['triplet_loss'])
        
        # 保存当前epoch的指标到日志
        self._save_epoch_metrics(epoch, train_metrics, val_metrics)
    
    def _save_epoch_metrics(self, epoch, train_metrics, val_metrics):
        """保存每个epoch的指标到日志文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}:\n")
            f.write("-" * 30 + "\n")
            
            for model_name in train_metrics.keys():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"Training:\n")
                f.write(f"  Loss: {train_metrics[model_name]['loss']:.4f}\n")
                f.write(f"  Accuracy: {train_metrics[model_name]['acc']:.2f}%\n")
                f.write(f"  CE Loss: {train_metrics[model_name]['ce_loss']:.4f}\n")
                f.write(f"  Triplet Loss: {train_metrics[model_name]['triplet_loss']:.4f}\n")
                
                f.write(f"Validation:\n")
                f.write(f"  Loss: {val_metrics[model_name]['loss']:.4f}\n")
                f.write(f"  Accuracy: {val_metrics[model_name]['acc']:.2f}%\n")
                f.write(f"  CE Loss: {val_metrics[model_name]['ce_loss']:.4f}\n")
                f.write(f"  Triplet Loss: {val_metrics[model_name]['triplet_loss']:.4f}\n")
    
    def plot_training_curves(self, save_name=None):
        """绘制训练曲线（损失和准确率）"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 获取当前epoch数（从任意一个非空模型获取）
        current_epochs = 0
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['train'][model_name]['loss']:
                current_epochs = len(self.metrics_history['train'][model_name]['loss'])
                break
                
        if current_epochs == 0:
            print("警告：没有可用的训练数据来绘制曲线")
            return
            
        epochs = range(current_epochs)
        
        # 训练损失
        ax1.set_title('Training Loss', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['train'][model_name]['loss']:
                ax1.plot(epochs, self.metrics_history['train'][model_name]['loss'], 
                        label=f'{model_name.upper()}', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 验证损失
        ax2.set_title('Validation Loss', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['val'][model_name]['loss']:
                ax2.plot(epochs, self.metrics_history['val'][model_name]['loss'], 
                        label=f'{model_name.upper()}', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # 训练准确率
        ax3.set_title('Training Accuracy', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['train'][model_name]['acc']:
                ax3.plot(epochs, self.metrics_history['train'][model_name]['acc'], 
                        label=f'{model_name.upper()}', linewidth=2)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        # 验证准确率
        ax4.set_title('Validation Accuracy', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['val'][model_name]['acc']:
                ax4.plot(epochs, self.metrics_history['val'][model_name]['acc'], 
                        label=f'{model_name.upper()}', linewidth=2)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_loss_components(self, save_name=None):
        """绘制损失组件（CE Loss和Triplet Loss）的变化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 获取当前epoch数（从任意一个非空模型获取）
        current_epochs = 0
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['train'][model_name]['ce_loss']:
                current_epochs = len(self.metrics_history['train'][model_name]['ce_loss'])
                break
                
        if current_epochs == 0:
            print("警告：没有可用的训练数据来绘制损失组件曲线")
            return
            
        epochs = range(current_epochs)
        
        # 训练CE Loss
        ax1.set_title('Training CE Loss', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['train'][model_name]['ce_loss']:
                ax1.plot(epochs, self.metrics_history['train'][model_name]['ce_loss'],
                        label=f'{model_name.upper()}', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('CE Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 验证CE Loss
        ax2.set_title('Validation CE Loss', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['val'][model_name]['ce_loss']:
                ax2.plot(epochs, self.metrics_history['val'][model_name]['ce_loss'],
                        label=f'{model_name.upper()}', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('CE Loss')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # 训练Triplet Loss
        ax3.set_title('Training Triplet Loss', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['train'][model_name]['triplet_loss']:
                ax3.plot(epochs, self.metrics_history['train'][model_name]['triplet_loss'],
                        label=f'{model_name.upper()}', linewidth=2)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Triplet Loss')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        # 验证Triplet Loss
        ax4.set_title('Validation Triplet Loss', fontsize=12, pad=10)
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['val'][model_name]['triplet_loss']:
                ax4.plot(epochs, self.metrics_history['val'][model_name]['triplet_loss'],
                        label=f'{model_name.upper()}', linewidth=2)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Triplet Loss')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = f'loss_components_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics_summary(self, save_name=None):
        """保存训练指标总结"""
        if save_name is None:
            save_name = f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(os.path.join(self.save_dir, save_name), 'w') as f:
            f.write("Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name in ['cbam', 'lgca']:
                if self.metrics_history['train'][model_name]['loss']:
                    f.write(f"{model_name.upper()} Model:\n")
                    f.write("-" * 30 + "\n")
                    
                    # 最佳验证准确率
                    best_val_acc = max(self.metrics_history['val'][model_name]['acc'])
                    best_val_epoch = self.metrics_history['val'][model_name]['acc'].index(best_val_acc) + 1
                    
                    # 最佳训练准确率
                    best_train_acc = max(self.metrics_history['train'][model_name]['acc'])
                    
                    # 最终性能
                    final_train_acc = self.metrics_history['train'][model_name]['acc'][-1]
                    final_val_acc = self.metrics_history['val'][model_name]['acc'][-1]
                    final_train_loss = self.metrics_history['train'][model_name]['loss'][-1]
                    final_val_loss = self.metrics_history['val'][model_name]['loss'][-1]
                    
                    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_val_epoch})\n")
                    f.write(f"Best Training Accuracy: {best_train_acc:.2f}%\n")
                    f.write(f"Final Training Accuracy: {final_train_acc:.2f}%\n")
                    f.write(f"Final Validation Accuracy: {final_val_acc:.2f}%\n")
                    f.write(f"Final Training Loss: {final_train_loss:.4f}\n")
                    f.write(f"Final Validation Loss: {final_val_loss:.4f}\n\n")
    
    def plot_learning_curves(self, save_name=None):
        """绘制学习曲线（训练vs验证）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 获取当前epoch数（从任意一个非空模型获取）
        current_epochs = 0
        for model_name in ['cbam', 'lgca']:
            if self.metrics_history['train'][model_name]['loss']:
                current_epochs = len(self.metrics_history['train'][model_name]['loss'])
                break
                
        if current_epochs == 0:
            print("警告：没有可用的训练数据来绘制学习曲线")
            return
            
        epochs = range(1, current_epochs + 1)
        
        # CBAM模型的学习曲线
        ax1.set_title('CBAM Learning Curves', fontsize=12, pad=10)
        if self.metrics_history['train']['cbam']['acc']:
            ax1.plot(epochs, self.metrics_history['train']['cbam']['acc'], 
                    label='Train Acc', linewidth=2)
            ax1.plot(epochs, self.metrics_history['val']['cbam']['acc'], 
                    label='Val Acc', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # LGCA模型的学习曲线
        ax2.set_title('LGCA Learning Curves', fontsize=12, pad=10)
        if self.metrics_history['train']['lgca']['acc']:
            ax2.plot(epochs, self.metrics_history['train']['lgca']['acc'], 
                    label='Train Acc', linewidth=2)
            ax2.plot(epochs, self.metrics_history['val']['lgca']['acc'], 
                    label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = f'learning_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close() 