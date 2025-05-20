"""
3D DenseNet Implementation for Cancer Stage Classification

This module provides a PyTorch implementation of 3D DenseNet for cancer stage classification
from CT scans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    """Basic building block of DenseNet: Dense Layer"""

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """DenseBlock: multiple dense layers stacked together"""

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    """Transition layer between DenseBlocks for downsampling"""

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    """3D DenseNet model for volumetric data classification"""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=4):
        """
        Args:
            growth_rate (int): How many filters to add each layer (k in paper)
            block_config (tuple): How many layers in each DenseBlock
            num_init_features (int): Number of filters in the first convolution layer
            bn_size (int): Multiplicative factor for number of bottleneck layers
            drop_rate (float): Dropout rate after each dense layer
            num_classes (int): Number of classification classes
        """
        super(DenseNet3D, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add dropout after each dense block
            if i > 0:
                self.features.add_module(f'dropout{i}', nn.Dropout3d(p=0.2))

            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # Global average pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),  # Keep high dropout before final classification
            nn.Linear(num_features, 512),  # Add an intermediate layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),  # Additional dropout
            nn.Linear(512, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


def DenseNet121_3D(num_classes=4, pretrained=False):
    """DenseNet-121 model for 3D data"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)


def DenseNet169_3D(num_classes=4, pretrained=False):
    """DenseNet-169 model for 3D data"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_classes)


def DenseNet201_3D(num_classes=4, pretrained=False):
    """DenseNet-201 model for 3D data"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=num_classes)


def DenseNet161_3D(num_classes=4, pretrained=False):
    """DenseNet-161 model for 3D data"""
    return DenseNet3D(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=num_classes)
