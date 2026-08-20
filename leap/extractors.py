import torch
import torch.nn as nn
import torchvision.models as tv


class VGG19FeatureExtractor(nn.Module):
    """ImageNet-pretrained VGG19 truncated before its classifier layer.

    Output is 4096-dimensional. `freeze_until` freezes the first N convolutional modules.
    """

    def __init__(self, freeze_until: int = 28):
        super().__init__()
        vgg = tv.vgg19(weights=tv.VGG19_Weights.DEFAULT)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1])
        for param in self.features[:freeze_until].parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet50FeatureExtractor(nn.Module):
    """ImageNet-pretrained ResNet50 truncated before its classifier layer.

    Output is 2048-dimensional. `freeze_until` freezes the first N child stages.
    """

    def __init__(self, freeze_until: int = 6):
        super().__init__()
        resnet = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        for i, child in enumerate(self.features.children()):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return torch.flatten(self.features(x), 1)


class DenseNet121FeatureExtractor(nn.Module):
    """ImageNet-pretrained DenseNet121 feature stack with global average pooling.

    Output is 1024-dimensional. `freeze_until` freezes the first N dense blocks.
    """

    def __init__(self, freeze_until: int = 2):
        super().__init__()
        densenet = tv.densenet121(weights=tv.DenseNet121_Weights.DEFAULT)
        self.features = densenet.features
        block_counter = 0
        for name, child in self.features.named_children():
            if "denseblock" in name:
                if block_counter < freeze_until:
                    for param in child.parameters():
                        param.requires_grad = False
                block_counter += 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
