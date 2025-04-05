import os
import shutil
from tqdm import tqdm

def organize_market1501(root_dir):
    """
    整理Market-1501数据集，将图片按ID分类存放
    
    Args:
        root_dir: Market-1501数据集根目录
    """
    # 处理训练集和验证集
    for split in ['train', 'val']:
        src_dir = os.path.join(root_dir, split)
        if not os.path.exists(src_dir):
            print(f"目录不存在: {src_dir}")
            continue
            
        print(f"\n处理{split}集...")
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # 使用tqdm显示进度条
        for img_file in tqdm(image_files, desc=f"整理{split}集"):
            # 获取行人ID（文件名前4位）
            person_id = img_file[:4]
            
            # 创建对应的ID目录
            id_dir = os.path.join(src_dir, person_id)
            os.makedirs(id_dir, exist_ok=True)
            
            # 移动图片到对应ID目录
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(id_dir, img_file)
            shutil.move(src_path, dst_path)

def main():
    # 从config.py读取数据集路径
    from config import Config
    cfg = Config()
    root_dir = cfg.data_dir
    
    print(f"开始整理数据集: {root_dir}")
    organize_market1501(root_dir)
    print("\n数据集整理完成！")
    print("目录结构已更新为：")
    print("market1501/")
    print("    └── train/")
    print("        ├── 0001/")
    print("        ├── 0002/")
    print("        └── ...")
    print("    └── val/")
    print("        ├── 0001/")
    print("        ├── 0002/")
    print("        └── ...")

if __name__ == '__main__':
    main() 