import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Callable, Optional, Tuple
import PIL.Image as Image

class FilteredImageFolder(datasets.ImageFolder):
    """过滤掉特定类别的ImageFolder数据集"""
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        # 获取所有类别
        valid_classes = [cls_name for cls_name in self.classes if cls_name != "0000"]
        valid_class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
        
        # 过滤样本
        valid_samples = []
        for path, target in self.samples:
            class_name = os.path.basename(os.path.dirname(path))
            if class_name != "0000":
                # 更新类别索引
                new_target = valid_class_to_idx[class_name]
                valid_samples.append((path, new_target))
        
        self.samples = valid_samples
        self.imgs = valid_samples
        self.classes = valid_classes
        self.class_to_idx = valid_class_to_idx
        self.targets = [s[1] for s in valid_samples]

def get_transforms(input_size=224):
    """获取数据预处理转换"""
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_val

def create_dataloaders(data_dir, batch_size, num_workers):
    """创建数据加载器"""
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = FilteredImageFolder(train_dir, transform=transform_train)
    val_dataset = FilteredImageFolder(val_dir, transform=transform_val)
    
    # 打印类别信息以进行调试
    print(f"找到的有效类别数量: {len(train_dataset.classes)}")
    print(f"类别到索引的映射: {train_dataset.class_to_idx}")
    print(f"训练集样本数量: {len(train_dataset)}")
    print(f"验证集样本数量: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 