import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cbam import ChannelAttention, SpatialAttention


# 保持原有的ChannelAttention、SpatialAttention、CBAM、Bottleneck类不变
# 新增LGCA模块
class LGCA(nn.Module):
    def __init__(self, in_planes, stripe_num=6):
        super(LGCA, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        self.stripe_num = stripe_num
        # 局部空间注意力分支
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_planes * stripe_num, in_planes // 4, 1),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU()
        )
        self.local_sa = SpatialAttention()
    def forward(self, x):
        # 全局分支
        x_global = self.ca(x) * x
        x_global = self.sa(x_global) * x_global
        # 局部分支
        b, c, h, w = x.size()
        stripe_h = h // self.stripe_num
        local_feat = [x[:, :, i * stripe_h:(i + 1) * stripe_h, :] for i in range(self.stripe_num)]
        local_feat = torch.cat([
            F.adaptive_avg_pool2d(part, (stripe_h, w))
            for part in local_feat], 1)
        local_feat = self.local_conv(local_feat)
        local_att = self.local_sa(local_feat)
        # 调整 local_att 的形状以匹配 x_global
        local_att = F.interpolate(local_att, size=(h, w), mode='bilinear', align_corners=False)
        # 特征融合
        return x_global + local_att * x_global


# 修改后的Bottleneck（集成LGCA）
class BottleneckLG(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_lgca=False):
        super(BottleneckLG, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_lgca = use_lgca
        if self.use_lgca:
            self.lgca = LGCA(planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.use_lgca:
            out = self.lgca(out)

        out += residual
        out = self.relu(out)

        return out


# 特征金字塔模块
class FeaturePyramid(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, features):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        return [self.fpn_convs[i](laterals[i]) for i in range(len(features))]


# 最终模型
class ResNet50_LGCA(nn.Module):
    def __init__(self, num_classes=1501, use_lgca=True):
        super(ResNet50_LGCA, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 使用修改后的BottleneckLG
        self.layer1 = self._make_layer(BottleneckLG, 64, 3, stride=1, use_lgca=use_lgca)
        self.layer2 = self._make_layer(BottleneckLG, 128, 4, stride=2, use_lgca=use_lgca)
        self.layer3 = self._make_layer(BottleneckLG, 256, 6, stride=2, use_lgca=use_lgca)
        self.layer4 = self._make_layer(BottleneckLG, 512, 3, stride=2, use_lgca=use_lgca)

        # 特征金字塔
        self.fpn = FeaturePyramid(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

        # 计算特征维度
        self.feature_dim = 2048  # layer4的输出通道数
        self.fc = nn.Linear(self.feature_dim, num_classes)

        # 添加钩子来获取梯度和特征图
        self.gradients = None
        self.features = None
        
        # 为最后一个卷积层注册钩子
        self.layer4.register_forward_hook(self.save_features)
        self.layer4.register_backward_hook(self.save_gradients)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_lgca=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_lgca))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_lgca=use_lgca))

        return nn.Sequential(*layers)

    def save_features(self, module, input, output):
        self.features = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def get_last_conv_gradients(self):
        return self.gradients
    
    def get_last_conv_features(self):
        return self.features

    def forward(self, x):
        # 主干网络特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 获取特征
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, H, W]

        # 全局平均池化
        features = self.avgpool(x)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        
        # 分类
        x = self.fc(features)
        
        return x, features
