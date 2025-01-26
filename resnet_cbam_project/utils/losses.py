import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """带标签平滑的交叉熵损失"""
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型预测结果 [batch_size, num_classes]
            targets: 真实标签 [batch_size]
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """带有硬样本挖掘的三元组损失"""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 特征向量 [batch_size, feat_dim]
            targets: 标签 [batch_size]
        """
        n = inputs.size(0)
        
        # 计算欧氏距离
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        
        # 获取正样本和负样本的mask
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 最难的正样本
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 最难的负样本
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # y=1 表示 dist_an > dist_ap + margin
        y = torch.ones_like(dist_an)
        
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CombinedLoss(nn.Module):
    """结合交叉熵损失和三元组损失"""
    def __init__(self, num_classes, margin=0.3, epsilon=0.1, lambda_ce=1.0, lambda_tri=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = CrossEntropyLabelSmooth(num_classes, epsilon)
        self.triplet_loss = TripletLoss(margin)
        self.lambda_ce = lambda_ce
        self.lambda_tri = lambda_tri

    def forward(self, outputs, features, targets):
        """
        Args:
            outputs: 分类输出 [batch_size, num_classes]
            features: 特征向量 [batch_size, feat_dim]
            targets: 标签 [batch_size]
        """
        loss_ce = self.ce_loss(outputs, targets)
        loss_tri = self.triplet_loss(features, targets)
        total_loss = self.lambda_ce * loss_ce + self.lambda_tri * loss_tri
        return total_loss, loss_ce, loss_tri 