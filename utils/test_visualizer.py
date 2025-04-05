import os
import numpy as np
from visualizer import TrainingVisualizer

def generate_dummy_metrics(num_epochs=10):
    """生成模拟的训练指标数据"""
    train_metrics = {
        'cbam': {
            'loss': np.random.uniform(0.5, 2.0, num_epochs),
            'acc': np.random.uniform(60, 95, num_epochs),
            'ce_loss': np.random.uniform(0.3, 1.5, num_epochs),
            'triplet_loss': np.random.uniform(0.2, 1.0, num_epochs)
        },
        'lgca': {
            'loss': np.random.uniform(0.5, 2.0, num_epochs),
            'acc': np.random.uniform(60, 95, num_epochs),
            'ce_loss': np.random.uniform(0.3, 1.5, num_epochs),
            'triplet_loss': np.random.uniform(0.2, 1.0, num_epochs)
        }
    }
    return train_metrics

def test_visualizer():
    """测试可视化器的所有功能"""
    print("开始测试可视化器...")
    
    # 创建测试目录
    test_dir = 'test_visualization'
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    
    # 初始化可视化器
    visualizer = TrainingVisualizer(save_dir=test_dir)
    print("✓ 可视化器初始化成功")
    
    # 生成模拟数据
    num_epochs = 10
    train_metrics = generate_dummy_metrics(num_epochs)
    val_metrics = generate_dummy_metrics(num_epochs)
    
    # 测试更新指标
    print("\n测试更新指标...")
    for epoch in range(num_epochs):
        current_train_metrics = {
            model: {metric: values[epoch] for metric, values in metrics.items()}
            for model, metrics in train_metrics.items()
        }
        current_val_metrics = {
            model: {metric: values[epoch] for metric, values in metrics.items()}
            for model, metrics in val_metrics.items()
        }
        visualizer.update_metrics(epoch, current_train_metrics, current_val_metrics)
    print("✓ 指标更新成功")
    
    # 测试绘制训练曲线
    print("\n测试绘制训练曲线...")
    visualizer.plot_training_curves('test_training_curves.png')
    print("✓ 训练曲线绘制成功")
    
    # 测试绘制损失组件
    print("\n测试绘制损失组件...")
    visualizer.plot_loss_components('test_loss_components.png')
    print("✓ 损失组件绘制成功")
    
    # 测试绘制学习曲线
    print("\n测试绘制学习曲线...")
    visualizer.plot_learning_curves('test_learning_curves.png')
    print("✓ 学习曲线绘制成功")
    
    # 测试保存指标总结
    print("\n测试保存指标总结...")
    visualizer.save_metrics_summary('test_summary.txt')
    print("✓ 指标总结保存成功")
    
    # 验证文件是否生成
    expected_files = [
        'test_training_curves.png',
        'test_loss_components.png',
        'test_learning_curves.png',
        'test_summary.txt',
        'training_log.txt'
    ]
    
    print("\n验证文件生成...")
    for file in expected_files:
        file_path = os.path.join(test_dir, visualizer.save_dir.split('/')[-1], file)
        if os.path.exists(file_path):
            print(f"✓ {file} 生成成功")
        else:
            print(f"✗ {file} 未生成")
    
    print("\n测试完成！")
    print(f"测试文件保存在: {os.path.join(test_dir, visualizer.save_dir.split('/')[-1])}")

if __name__ == '__main__':
    test_visualizer() 