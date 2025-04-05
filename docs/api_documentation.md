# 行人重识别系统 API 文档

## 一、模型接口

### 1.1 CBAM模型

```python
class CBAM(nn.Module):
    def __init__(self, num_classes=751, pretrained=True):
        """
        初始化CBAM模型
        
        参数:
            num_classes (int): 类别数量
            pretrained (bool): 是否使用预训练权重
        """
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入图像张量
            
        返回:
            tuple: (分类分数, 特征向量)
        """
```

### 1.2 LGCA模型

```python
class LGCA(nn.Module):
    def __init__(self, num_classes=751, pretrained=True):
        """
        初始化LGCA模型
        
        参数:
            num_classes (int): 类别数量
            pretrained (bool): 是否使用预训练权重
        """
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入图像张量
            
        返回:
            tuple: (分类分数, 特征向量)
        """
```

## 二、训练接口

### 2.1 训练器

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        """
        初始化训练器
        
        参数:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
        """
        
    def train(self, num_epochs):
        """
        训练模型
        
        参数:
            num_epochs (int): 训练轮数
        """
        
    def validate(self):
        """
        验证模型
        
        返回:
            dict: 验证指标
        """
```

## 三、可视化接口

### 3.1 训练可视化器

```python
class TrainingVisualizer:
    def __init__(self, save_dir='visualization'):
        """
        初始化可视化器
        
        参数:
            save_dir (str): 保存目录
        """
        
    def update_metrics(self, epoch, train_metrics, val_metrics):
        """
        更新训练指标
        
        参数:
            epoch (int): 当前轮数
            train_metrics (dict): 训练指标
            val_metrics (dict): 验证指标
        """
        
    def plot_training_curves(self, save_name=None):
        """
        绘制训练曲线
        
        参数:
            save_name (str): 保存文件名
        """
```

## 四、数据接口

### 4.1 数据集组织器

```python
class DatasetOrganizer:
    def __init__(self, root_dir):
        """
        初始化数据集组织器
        
        参数:
            root_dir (str): 数据集根目录
        """
        
    def organize(self):
        """
        组织数据集结构
        """
        
    def get_data_loaders(self, batch_size=32):
        """
        获取数据加载器
        
        参数:
            batch_size (int): 批次大小
            
        返回:
            tuple: (训练加载器, 验证加载器)
        """
```

## 五、配置接口

### 5.1 配置类

```python
class Config:
    # 数据集配置
    DATASET_ROOT = 'market1501'
    TRAIN_BATCH_SIZE = 32
    
    # 模型配置
    MODEL_NAME = 'cbam'
    NUM_CLASSES = 751
    
    # 训练配置
    LEARNING_RATE = 0.0003
    NUM_EPOCHS = 120
```

## 六、Web接口

### 6.1 演示系统

```python
class ReIDDemo:
    def __init__(self, model_path):
        """
        初始化演示系统
        
        参数:
            model_path (str): 模型路径
        """
        
    def process_image(self, image):
        """
        处理输入图像
        
        参数:
            image: 输入图像
            
        返回:
            dict: 处理结果
        """
```

## 七、使用示例

### 7.1 模型训练

```python
# 初始化模型
model = CBAM(num_classes=751)

# 初始化训练器
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)

# 开始训练
trainer.train(num_epochs=120)
```

### 7.2 模型推理

```python
# 初始化演示系统
demo = ReIDDemo(model_path='checkpoints/best_model.pth')

# 处理图像
result = demo.process_image(image)
```

### 7.3 可视化训练过程

```python
# 初始化可视化器
visualizer = TrainingVisualizer(save_dir='visualization')

# 更新指标
visualizer.update_metrics(epoch, train_metrics, val_metrics)

# 绘制训练曲线
visualizer.plot_training_curves()
``` 