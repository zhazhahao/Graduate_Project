import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class AttentionVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()  # 设置为评估模式
        
    def generate_attention_map(self, image, target_class=None):
        """
        生成注意力热力图
        Args:
            image: 输入图像 (PIL Image)
            target_class: 目标类别，如果为None则使用预测类别
        Returns:
            attention_map: 注意力热力图
            predicted_class: 预测的类别
        """
        # 预处理图像
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image = image.to(self.device)
        
        # 获取模型输出
        output, _ = self.model(image)  # 解包输出，忽略特征
        
        # 如果没有指定目标类别，使用预测的类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 清除梯度
        self.model.zero_grad()
        
        # 计算目标类别的梯度
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        # 获取最后一个卷积层的梯度
        gradients = self.model.get_last_conv_gradients()
        pooled_gradients = torch.mean(gradients, dim=[2, 3])
        
        # 获取最后一个卷积层的特征图
        features = self.model.get_last_conv_features()
        
        # 计算权重
        for i in range(features.size(1)):
            features[:, i, :, :] *= pooled_gradients[:, i]
        
        # 生成热力图
        attention_map = torch.mean(features, dim=1).detach().cpu()
        attention_map = F.interpolate(attention_map.unsqueeze(0), 
                                    size=(224, 224), 
                                    mode='bilinear', 
                                    align_corners=False)
        attention_map = attention_map.squeeze().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        return attention_map, target_class
    
    def visualize_attention(self, image, attention_map, save_path=None):
        """
        可视化注意力热力图
        Args:
            image: 原始图像 (PIL Image)
            attention_map: 注意力热力图
            save_path: 保存路径
        """
        # 转换图像格式
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 调整热力图大小以匹配图像
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # 创建热力图
        attention_map_uint8 = np.uint8(255 * attention_map)
        heatmap = cv2.applyColorMap(attention_map_uint8, cv2.COLORMAP_HOT)
        
        # 创建二值化的注意力重点图
        _, binary_map = cv2.threshold(attention_map_uint8, 150, 255, cv2.THRESH_BINARY)
        binary_map = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
        
        # 叠加热力图和原始图像
        output = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        # 显示或保存结果
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')
        
        # 热力图叠加
        plt.subplot(1, 3, 2)
        plt.imshow(output)
        plt.title('热力图叠加')
        plt.axis('off')
        
        # 注意力重点
        plt.subplot(1, 3, 3)
        plt.imshow(binary_map)
        plt.title('注意力重点')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 