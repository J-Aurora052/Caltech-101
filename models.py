import torch
import torch.nn as nn
from torchvision import models


class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes=101, pretrained=True):
        super(FineTunedResNet18, self).__init__()
        # 加载预训练的ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # 冻结所有层
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 替换最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # 解冻最后的全连接层
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)

    def get_pretrained_params(self):
        # 返回预训练层的参数（不包括最后的全连接层）
        return [param for name, param in self.resnet.named_parameters() if 'fc' not in name]

    def get_new_params(self):
        # 返回新添加层的参数
        return self.resnet.fc.parameters()