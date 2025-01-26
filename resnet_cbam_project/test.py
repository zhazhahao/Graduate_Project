import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.resnet_cbam import ResNet50_CBAM
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score


def extract_features(model, dataloader):
    """提取特征向量"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Extracting features'):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            outputs, feat = model(inputs)
            features.append(feat.cpu().numpy())
            labels.append(targets.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def compute_distance_matrix(query_features, gallery_features):
    """计算查询集和图库集之间的距离矩阵"""
    query_features = torch.from_numpy(query_features)
    gallery_features = torch.from_numpy(gallery_features)
    
    m, n = query_features.size(0), gallery_features.size(0)
    dist_mat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
    
    return dist_mat.numpy()


def evaluate(dist_mat, query_labels, gallery_labels, max_rank=50):
    """评估ReID性能"""
    num_query = dist_mat.shape[0]
    
    # 计算CMC和mAP
    all_cmc = []
    all_AP = []
    num_valid_query = 0
    
    for q_idx in range(num_query):
        # 获取查询图片的标签
        q_label = query_labels[q_idx]
        
        # 从距离矩阵中移除查询图片本身
        dist = dist_mat[q_idx]
        
        # 获取正样本和负样本的索引
        pos_idx = np.where(gallery_labels == q_label)[0]
        neg_idx = np.where(gallery_labels != q_label)[0]
        
        if len(pos_idx) == 0:
            continue
            
        # 按距离排序
        indices = np.argsort(dist)
        matches = np.in1d(indices, pos_idx)
        
        # 计算CMC
        raw_cmc = matches[0:max_rank]
        if not np.any(raw_cmc):
            continue
            
        cmc = np.cumsum(raw_cmc)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)
        
        # 计算AP
        num_rel = len(pos_idx)
        tmp_cmc = np.binary_search(indices, pos_idx)
        tmp_cmc = np.cumsum(tmp_cmc)
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
        num_valid_query += 1
        
    assert num_valid_query > 0, "No valid query"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_query
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP


def main(args):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    query_dataset = ImageFolder(os.path.join(args.data_dir, 'query'), transform=transform)
    gallery_dataset = ImageFolder(os.path.join(args.data_dir, 'gallery'), transform=transform)

    query_loader = DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 加载模型
    model = ResNet50_CBAM(num_classes=args.num_classes, use_cbam=True)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    if torch.cuda.is_available():
        model = model.cuda()

    # 提取特征
    print("Extracting features...")
    query_features, query_labels = extract_features(model, query_loader)
    gallery_features, gallery_labels = extract_features(model, gallery_loader)

    # 计算距离矩阵
    print("Computing distance matrix...")
    dist_mat = compute_distance_matrix(query_features, gallery_features)

    # 评估性能
    print("Evaluating...")
    cmc, mAP = evaluate(dist_mat, query_labels, gallery_labels)

    # 打印结果
    print(f"Results ----------")
    print(f"mAP: {mAP:.1%}")
    print(f"CMC curve")
    for r in [1, 5, 10]:
        print(f"Rank-{r}: {cmc[r-1]:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ReID Model')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='test batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='number of classes in the dataset')

    args = parser.parse_args()
    main(args) 