import os
import torch

def save_checkpoint(state, save_dir, is_best=False, filename='checkpoint.pth'):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_filepath)

def load_checkpoint(resume_path, models, optimizers, schedulers):
    """加载检查点"""
    if not os.path.isfile(resume_path):
        return 0, {model_name: 0.0 for model_name in models.keys()}, 0
    
    print(f'Loading checkpoint {resume_path}')
    checkpoint = torch.load(resume_path)
    
    start_epoch = checkpoint['epoch'] + 1
    
    # 初始化best_acc字典
    best_acc = {model_name: 0.0 for model_name in models.keys()}
    
    # 从检查点加载最佳准确率
    if 'best_acc' in checkpoint:
        best_acc.update(checkpoint['best_acc'])
    
    # 根据实际启用的模型加载参数
    if 'cbam' in models:
        if 'model_cbam_state_dict' in checkpoint:
            models['cbam'].load_state_dict(checkpoint['model_cbam_state_dict'])
            optimizers['cbam'].load_state_dict(checkpoint['optimizer_cbam_state_dict'])
            schedulers['cbam'].load_state_dict(checkpoint['scheduler_cbam_state_dict'])
            print('已加载CBAM模型参数')
        else:
            print('警告：检查点中未找到CBAM模型参数')
    
    if 'lgca' in models:
        if 'model_lgca_state_dict' in checkpoint:
            models['lgca'].load_state_dict(checkpoint['model_lgca_state_dict'])
            optimizers['lgca'].load_state_dict(checkpoint['optimizer_lgca_state_dict'])
            schedulers['lgca'].load_state_dict(checkpoint['scheduler_lgca_state_dict'])
            print('已加载LGCA模型参数')
        else:
            print('警告：检查点中未找到LGCA模型参数')
    
    print(f'Loaded checkpoint from epoch {start_epoch-1}')
    return start_epoch, best_acc, 0  # 最后一个参数不再使用 