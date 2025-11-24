import torch
import torch.nn as nn


class MultiScaleConv(nn.Module):
    """
    多尺度卷积模块，使用不同核大小的卷积并行提取特征
    """

    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv7x7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 并行多尺度卷积
        feat_3x3 = self.conv3x3(x)
        feat_5x5 = self.conv5x5(x)
        feat_7x7 = self.conv7x7(x)
        # 特征融合
        feats = torch.stack([feat_3x3, feat_5x5, feat_7x7], dim=2)  # [B, C, 3, H, W]
        # 通道维度softmax加权
        attn_weights = self.softmax(feats.mean(dim=[3, 4]))  # [B, C, 3]
        attn_weights = attn_weights.view(b, c, 3, 1, 1)
        # 加权融合
        out = (feats * attn_weights).sum(dim=2)
        return out


class ChannelAttention(nn.Module):
    """
    多尺度通道注意力模块，替换传统全局池化为多尺度卷积
    """

    def __init__(self, channels):
        super().__init__()
        self.multi_scale_conv = MultiScaleConv(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 多尺度特征提取作为通道权重
        scale_attn = self.multi_scale_conv(x)
        return x * self.sigmoid(scale_attn)


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度的平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接特征
        feat = torch.cat([avg_out, max_out], dim=1)
        # 空间注意力权重
        spatial_attn = self.sigmoid(self.conv(feat))
        return x * spatial_attn


class MSAM(nn.Module):
    """
    多尺度注意力模块 - 结合多尺度通道注意力和空间注意力
    """

    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 先应用通道注意力
        x = self.channel_attention(x)
        # 再应用空间注意力
        x = self.spatial_attention(x)
        return x
