import torch
import torch.nn as nn
import torchvision.models as models
    
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, freeze_until=28):
        super(VGG19FeatureExtractor, self).__init__()
        
        # Load VGG19 model with the most up-to-date pretrained weights
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1])  # Remove the last classification layer
        
        # Freeze layers up to `freeze_until`
        for param in self.features[:freeze_until].parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x # Output is 4096 Dimensional
    
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, freeze_until=6):
        super(ResNet50FeatureExtractor, self).__init__()
        
        # Load ResNet50 model with the most up-to-date pretrained weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last classification layer
        
        # Freeze layers up to `freeze_until` stage
        # ResNet has 7 stages, so freeze_until=6 freezes up to the final layer before the average pooling
        child_counter = 0
        for child in self.features.children():
            if child_counter < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False
            child_counter += 1

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x # Output is 2048 dimensional
        
        
class DenseNet121FeatureExtractor(nn.Module):
    def __init__(self, freeze_until=2):
        super(DenseNet121FeatureExtractor, self).__init__()
        
        # Load DenseNet-121 model with pretrained weights
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        self.features = densenet.features  # The main DenseNet feature extractor without the classifier
        
        # Freeze the first `freeze_until` dense blocks
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
        x = torch.flatten(x, 1)
        return x  # Output is 1024-dimensional
        
        