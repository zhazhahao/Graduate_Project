import torch
from torch.utils.tensorboard import SummaryWriter
from .metrics import AverageMeter, compute_loss_accuracy
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, models, criterion, optimizers, schedulers, device, model_config):
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.model_config = model_config
        
        # 只初始化启用的模型
        if model_config['cbam']['enabled']:
            self.models['cbam'] = models['cbam']
            self.optimizers['cbam'] = optimizers['cbam']
            self.schedulers['cbam'] = schedulers['cbam']
            
        if model_config['lgca']['enabled']:
            self.models['lgca'] = models['lgca']
            self.optimizers['lgca'] = optimizers['lgca']
            self.schedulers['lgca'] = schedulers['lgca']
        
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter('runs/experiment')

    def train_epoch(self, epoch, train_loader):
        """训练一个epoch"""
        metrics = {}
        
        for model_name, model in self.models.items():
            model.train()
            losses = AverageMeter()
            ce_losses = AverageMeter()
            triplet_losses = AverageMeter()
            top1 = AverageMeter()
            
            # 创建进度条
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training {model_name.upper()}')
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 训练模型
                self.optimizers[model_name].zero_grad()
                outputs, features = model(inputs)
                
                # 计算组合损失
                loss, ce_loss, triplet_loss = self.criterion(outputs, features, targets)
                loss.backward()
                self.optimizers[model_name].step()
                
                # 计算准确率
                _, acc1, _ = compute_loss_accuracy(outputs, targets, self.criterion.ce_criterion)
                
                # 更新统计
                losses.update(loss.item(), inputs.size(0))
                ce_losses.update(ce_loss.item(), inputs.size(0))
                triplet_losses.update(triplet_loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'ce_loss': f'{ce_losses.avg:.4f}',
                    'triplet_loss': f'{triplet_losses.avg:.4f}',
                    'acc': f'{top1.avg:.2f}%'
                })
            
            # 记录训练指标
            self.writer.add_scalar(f'Train/Total_Loss_{model_name.upper()}', losses.avg, epoch)
            self.writer.add_scalar(f'Train/CE_Loss_{model_name.upper()}', ce_losses.avg, epoch)
            self.writer.add_scalar(f'Train/Triplet_Loss_{model_name.upper()}', triplet_losses.avg, epoch)
            self.writer.add_scalar(f'Train/Acc_{model_name.upper()}', top1.avg, epoch)
            
            metrics[model_name] = {
                'loss': losses.avg,
                'ce_loss': ce_losses.avg,
                'triplet_loss': triplet_losses.avg,
                'acc': top1.avg
            }
        
        return metrics

    def validate(self, epoch, val_loader):
        """验证模型"""
        metrics = {}
        
        for model_name, model in self.models.items():
            model.eval()
            losses = AverageMeter()
            ce_losses = AverageMeter()
            triplet_losses = AverageMeter()
            top1 = AverageMeter()
            
            # 创建进度条
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Validating {model_name.upper()}')
            
            with torch.no_grad():
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs, features = model(inputs)
                    loss, ce_loss, triplet_loss = self.criterion(outputs, features, targets)
                    
                    # 计算准确率
                    _, acc1, _ = compute_loss_accuracy(outputs, targets, self.criterion.ce_criterion)
                    
                    # 更新统计
                    losses.update(loss.item(), inputs.size(0))
                    ce_losses.update(ce_loss.item(), inputs.size(0))
                    triplet_losses.update(triplet_loss.item(), inputs.size(0))
                    top1.update(acc1.item(), inputs.size(0))
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{losses.avg:.4f}',
                        'ce_loss': f'{ce_losses.avg:.4f}',
                        'triplet_loss': f'{triplet_losses.avg:.4f}',
                        'acc': f'{top1.avg:.2f}%'
                    })
            
            # 记录验证指标
            self.writer.add_scalar(f'Val/Total_Loss_{model_name.upper()}', losses.avg, epoch)
            self.writer.add_scalar(f'Val/CE_Loss_{model_name.upper()}', ce_losses.avg, epoch)
            self.writer.add_scalar(f'Val/Triplet_Loss_{model_name.upper()}', triplet_losses.avg, epoch)
            self.writer.add_scalar(f'Val/Acc_{model_name.upper()}', top1.avg, epoch)
            
            metrics[model_name] = {
                'loss': losses.avg,
                'ce_loss': ce_losses.avg,
                'triplet_loss': triplet_losses.avg,
                'acc': top1.avg
            }
        
        return metrics 