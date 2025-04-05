import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """三元组损失"""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CombinedLoss(nn.Module):
    """组合损失：交叉熵损失 + 三元组损失"""
    def __init__(self, ce_weight=1.0, triplet_weight=1.0, margin=0.3):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.margin = margin
        self.ce_criterion = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=margin)
    
    def forward(self, outputs, features, targets):
        """
        Args:
            outputs: 分类输出 [batch_size, num_classes]
            features: 特征向量 [batch_size, feature_dim]
            targets: 标签 [batch_size]
        """
        # 计算交叉熵损失
        ce_loss = self.ce_criterion(outputs, targets)
        
        # 为每个样本找到正负样本对
        triplet_loss = self._get_triplet_loss(features, targets)
        
        # 组合损失
        total_loss = self.ce_weight * ce_loss + self.triplet_weight * triplet_loss
        
        return total_loss, ce_loss, triplet_loss
    
    def _get_triplet_loss(self, features, targets):
        """为每个样本构建三元组并计算损失"""
        batch_size = features.size(0)
        if batch_size < 3:  # 需要至少3个样本才能构建三元组
            return torch.tensor(0.0, device=features.device)
            
        # 计算特征之间的欧氏距离
        dist_matrix = torch.cdist(features, features)
        
        # 获取正负样本对的mask
        pos_mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        neg_mask = ~pos_mask
        
        triplet_loss = torch.tensor(0.0, device=features.device)
        
        for i in range(batch_size):
            pos_dist = dist_matrix[i][pos_mask[i]]  # 正样本距离
            neg_dist = dist_matrix[i][neg_mask[i]]  # 负样本距离
            
            if len(pos_dist) == 0 or len(neg_dist) == 0:
                continue
                
            # 选择最难的正样本（最远的）和最难的负样本（最近的）
            hardest_pos_dist = pos_dist.max()
            hardest_neg_dist = neg_dist.min()
            
            # 计算三元组损失
            triplet_loss += F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        return triplet_loss / batch_size if batch_size > 0 else triplet_loss 