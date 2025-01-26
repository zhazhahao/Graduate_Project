import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.resnet_cbam import ResNet50_CBAM
from utils.losses import CombinedLoss
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm


def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 创建模型
    model = ResNet50_CBAM(num_classes=len(train_dataset.classes), use_cbam=True)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    model = model.cuda() if torch.cuda.is_available() else model

    # 定义损失函数和优化器
    criterion = CombinedLoss(
        num_classes=len(train_dataset.classes),
        margin=args.margin,
        epsilon=args.epsilon,
        lambda_ce=args.lambda_ce,
        lambda_tri=args.lambda_tri
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # TensorBoard
    writer = SummaryWriter(args.logs_dir)

    # 开始训练
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            train_loader, model, criterion, optimizer, epoch, writer
        )

        # 验证
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, writer)

        scheduler.step()

        # 保存最佳模型
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        
        filename = os.path.join(args.checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(state, filename)
        if is_best:
            best_file = os.path.join(args.checkpoints_dir, 'model_best.pth')
            torch.save(state, best_file)

        print(f'Epoch: [{epoch+1}/{args.epochs}]\t'
              f'Train Loss: {train_loss:.4f}\t'
              f'Train Acc: {train_acc:.2f}%\t'
              f'Val Loss: {val_loss:.4f}\t'
              f'Val Acc: {val_acc:.2f}%')

    writer.close()


def train_epoch(train_loader, model, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        # 前向传播
        outputs, features = model(inputs)
        loss, loss_ce, loss_tri = criterion(outputs, features, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 记录到TensorBoard
        step = epoch * len(train_loader) + i
        writer.add_scalar('Train/Loss', loss.item(), step)
        writer.add_scalar('Train/CE_Loss', loss_ce.item(), step)
        writer.add_scalar('Train/Triplet_Loss', loss_tri.item(), step)

    acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc


def validate(val_loader, model, criterion, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(val_loader, desc='Validation')):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # 前向传播
            outputs, features = model(inputs)
            loss, _, _ = criterion(outputs, features, targets)

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算准确率和平均损失
    acc = 100. * correct / total
    avg_loss = total_loss / len(val_loader)

    # 记录到TensorBoard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Accuracy', acc, epoch)

    return avg_loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ReID Model')
    # 数据集参数
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='path to dataset')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='mini-batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--lr-step', type=int, default=20,
                        help='step size for learning rate decay')
    
    # 损失函数参数
    parser.add_argument('--margin', type=float, default=0.3,
                        help='margin for triplet loss')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='smoothing epsilon for label smooth')
    parser.add_argument('--lambda-ce', type=float, default=1.0,
                        help='weight for cross entropy loss')
    parser.add_argument('--lambda-tri', type=float, default=1.0,
                        help='weight for triplet loss')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint')
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--logs-dir', type=str, default='./logs',
                        help='path to tensorboard logs')

    args = parser.parse_args()

    # 创建必要的目录
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    train(args) 