import torch
from PIL import Image
import os
from config import Config
from models.resnet_lgca import ResNet50_LGCA
from utils.visualization import AttentionVisualizer

def main():
    # 加载配置
    cfg = Config()
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = ResNet50_LGCA(num_classes=cfg.num_classes).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load(cfg.resume)
    model.load_state_dict(checkpoint['model_lgca_state_dict'])
    model.eval()
    
    # 创建可视化器
    visualizer = AttentionVisualizer(model, device)
    
    # 创建保存目录
    save_dir = os.path.join(cfg.save_dir, 'attention_maps')
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载测试图像
    test_image_path = r'D:\pycharmprojects\graduate_project\market1501\train\0007\0007_c2s3_070952_01.jpg'  # 替换为您的测试图像路径
    image = Image.open(test_image_path)
    
    # 生成并可视化注意力热力图
    attention_map, predicted_class = visualizer.generate_attention_map(image)
    save_path = os.path.join(save_dir, 'attention_map.jpg')
    visualizer.visualize_attention(image, attention_map, save_path)
    
    print(f'热力图已保存到: {save_path}')
    print(f'预测类别: {predicted_class}')

if __name__ == '__main__':
    main() 