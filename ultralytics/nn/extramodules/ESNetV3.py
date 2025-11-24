import torch
import torch.nn as nn

class SELayerV3(nn.Module):
    def __init__(self, channel, reduction=4, cardinality=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # 多分支Dense Layer
        self.dense_layers = nn.ModuleList()
        for _ in range(cardinality):
            self.dense_layers.append(nn.Sequential(
                nn.Linear(channel // reduction, channel // reduction, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.fc2 = nn.Linear(channel // reduction * cardinality, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        # 多分支特征融合
        dense_outs = []
        for layer in self.dense_layers:
            dense_outs.append(layer(y))
        y = torch.cat(dense_outs, dim=1)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ESNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.se_v3_1 = SELayerV3(128, reduction=4, cardinality=4)
        self.se_v3_2 = SELayerV3(256, reduction=4, cardinality=4)
        self.se_v3_3 = SELayerV3(512, reduction=4, cardinality=4)

    def forward(self, x):
        x = self.se_v3_1(x)
        x = self.se_v3_2(x)
        x = self.se_v3_3(x)
        return x
