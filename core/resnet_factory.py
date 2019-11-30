from __future__ import absolute_import
import math
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0), -1)
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0), 1, h, w)
        return z


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base(pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)

        # For Normal Baseline
        # resnet = ResNet.__factory[depth](pretrained=pretrained)
        # resnet.layer4[0].conv2.stride = (1, 1)
        # resnet.layer4[0].downsample[0].stride = (1, 1)
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        # For Spatial Attention
        self.resnet = ResNet.__factory[depth](pretrained=pretrained)
        self.resnet.layer4[0].conv2.stride = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.maxpool,  # no relu
            self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4)

        self.gap = nn.AdaptiveAvgPool2d(1)


        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            # For Normal Baseline
            # out_planes = resnet.fc.in_features

            # For Spatial Attention
            out_planes = self.resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        # For Spatial Attention
        self.SA = SpatialAttention()
        self.local_conv_layer1 = nn.Conv2d(256, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_conv_layer2 = nn.Conv2d(512, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_conv_layer3 = nn.Conv2d(1024, self.num_features, kernel_size=1, padding=0, bias=False)

        if not pretrained:
            self.reset_params()


    def forward(self, x, feature_withbn=True):

        # Normal Baseline
        # x = self.base(x)
        #
        # x = self.gap(x)
        # x = x.view(x.size(0), -1)
        #
        # if self.cut_at_pooling:
        #     return x
        #
        # if self.has_embedding:
        #     bn_x = self.feat_bn(self.feat(x))
        # else:
        #     bn_x = self.feat_bn(x)
        #
        # if self.training is False:
        #     bn_x = F.normalize(bn_x)
        #     return bn_x
        #
        # if self.norm:
        #     bn_x = F.normalize(bn_x)
        # elif self.has_embedding:
        #     bn_x = F.relu(bn_x)
        #
        # if self.dropout > 0:
        #     bn_x = self.drop(bn_x)
        #
        # if self.num_classes > 0:
        #     prob = self.classifier(bn_x)
        # else:
        #     return x, bn_x
        #
        # if feature_withbn:
        #     return bn_x, prob
        #
        # return x, prob


        # Spatial Attention
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.maxpool(x)
        x_layer1 = self.resnet.layer1(x)
        x_layer2 = self.resnet.layer2(x_layer1)
        x_layer3 = self.resnet.layer3(x_layer2)
        x_layer4 = self.resnet.layer4(x_layer3)

        x_attn1 = self.SA(x_layer1)
        x_attn2 = self.SA(x_layer2)
        x_attn3 = self.SA(x_layer3)

        x_layer1 = x_layer1 * x_attn1
        x_layer2 = x_layer2 * x_attn2
        x_layer3 = x_layer3 * x_attn3

        x_layer1 = self.gap(x_layer1)
        x_layer1 = self.local_conv_layer1(x_layer1)
        x_layer1 = x_layer1.view(x_layer1.size(0), -1)
        x_layer1 = self.feat_bn(x_layer1)
        x_layer1 = self.classifier(x_layer1)

        x_layer2 = self.gap(x_layer2)
        x_layer2 = self.local_conv_layer2(x_layer2)
        x_layer2 = x_layer2.view(x_layer2.size(0), -1)
        x_layer2 = self.feat_bn(x_layer2)
        x_layer2 = self.classifier(x_layer2)

        x_layer3 = self.gap(x_layer3)
        x_layer3 = self.local_conv_layer3(x_layer3)
        x_layer3 = x_layer3.view(x_layer3.size(0), -1)
        x_layer3 = self.feat_bn(x_layer3)
        x_layer3 = self.classifier(x_layer3)

        x_layer4 = self.gap(x_layer4)
        x_layer4 = x_layer4.view(x_layer4.size(0), -1)
        features = self.feat_bn(x_layer4)
        cls_score = self.classifier(features)

        return features, (x_layer1, x_layer2, x_layer3, cls_score)


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)



def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


def resnet_ibn50a(**kwargs):
    return ResNet('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNet('101a', **kwargs)

