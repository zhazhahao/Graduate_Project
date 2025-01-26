# ResNet50-CBAM 行人重识别项目

这是一个基于ResNet50和CBAM注意力机制的行人重识别（Person Re-identification）项目。该项目使用PyTorch实现，结合了交叉熵损失和三元组损失来训练模型。

## 项目结构

```
resnet_cbam_project/
├── data/                    # 数据集目录
│   ├── train/              # 训练集
│   ├── val/                # 验证集
│   ├── query/              # 测试集查询图片
│   └── gallery/            # 测试集图库图片
├── models/                  # 模型定义
│   └── resnet_cbam.py      # ResNet50-CBAM模型
├── utils/                   # 工具函数
│   └── losses.py           # 损失函数定义
├── configs/                 # 配置文件目录
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明
```

## 环境要求

- Python 3.7+
- PyTorch 1.8.0+
- CUDA（推荐）

详细的依赖要求请参见 `requirements.txt`。

## 安装

1. 克隆项目：
```bash
git clone https://github.com/your-username/resnet_cbam_project.git
cd resnet_cbam_project
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

1. 将数据集按照以下结构组织：
```
data/
├── train/
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── person2/
│       ├── image1.jpg
│       └── image2.jpg
├── val/
│   └── ...
├── query/
│   └── ...
└── gallery/
    └── ...
```

2. 每个人的图片应放在以其ID命名的单独文件夹中。

## 训练

使用以下命令开始训练：

```bash
python train.py --data-dir ./data \
                --batch-size 32 \
                --epochs 60 \
                --lr 3e-4 \
                --margin 0.3
```

主要参数说明：
- `--data-dir`: 数据集路径
- `--batch-size`: 批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--margin`: 三元组损失边界值

## 测试

使用以下命令评估模型：

```bash
python test.py --data-dir ./data \
               --checkpoint ./checkpoints/model_best.pth \
               --num-classes 1000
```

主要参数说明：
- `--data-dir`: 数据集路径
- `--checkpoint`: 模型检查点路径
- `--num-classes`: 数据集中的类别数（人数）

## 模型说明

本项目使用了以下技术：

1. **基础网络**: ResNet50
2. **注意力机制**: CBAM (Convolutional Block Attention Module)
3. **损失函数**: 
   - 带标签平滑的交叉熵损失
   - 带硬样本挖掘的三元组损失

## 性能指标

模型评估使用以下指标：
- Rank-1/5/10 准确率
- mAP (mean Average Precision)

## 引用

如果您使用了本项目的代码，请引用以下论文：

```
@inproceedings{woo2018cbam,
  title={CBAM: Convolutional Block Attention Module},
  author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。 