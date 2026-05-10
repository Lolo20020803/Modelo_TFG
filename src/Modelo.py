import numpy as np
import torch.nn as nn
import torch
BatchNorm = nn.BatchNorm2d

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.project = None
        if in_channels != out_channels or stride != 1:
            self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.project is not None:
            residual = self.project(residual)
        out += residual
        out = self.relu(out)
        return out


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class Feature_Aggregator(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        self.deconv = Deconv(in_channels_2, out_channels)
        self.block_1 = BasicBlock(in_channels_1 + out_channels, out_channels)
        self.block_2 = BasicBlock(out_channels, out_channels)
    def forward(self, x1, x2):
        x2 = self.deconv(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.block_1(x)
        x = self.block_2(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        residual = self.project(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Feature_Extractor(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=6, down_sample_input=False):
        super().__init__()
        self.down_sample_input = down_sample_input
        self.down_sample = DownSample(in_channels, out_channels) if down_sample_input else None

        blocks_modules = []
        if not down_sample_input:
            blocks_modules.append(BasicBlock(in_channels, out_channels))
        for _ in range(num_blocks - (1 if not down_sample_input else 0)):
            blocks_modules.append(BasicBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks_modules)

    def forward(self, x):
        if self.down_sample_input:
            x = self.down_sample(x)
        return self.blocks(x)

class Deep_Aggregation(nn.Module):
    def __init__(self, num_inputs, channels, num_outputs):
        super().__init__()
        self.extract_1a = Feature_Extractor(num_inputs, channels[0])
        self.extract_2a = Feature_Extractor(channels[0], channels[1], down_sample_input=True)
        self.extract_3a = Feature_Extractor(channels[1], channels[2], down_sample_input=True)
        self.aggregate_1b = Feature_Aggregator(channels[0], channels[1], channels[1])
        self.aggregate_2b = Feature_Aggregator(channels[1], channels[2], channels[2])
        self.aggregate_1c = Feature_Aggregator(channels[1], channels[2], channels[2])
        self.conv_1x1 = nn.Conv2d(channels[2], num_outputs, kernel_size=1)
    def forward(self, x):
        x_1a = self.extract_1a(x)
        x_2a = self.extract_2a(x_1a)
        x_3a = self.extract_3a(x_2a)
        x_1b = self.aggregate_1b(x_1a, x_2a)
        x_2b = self.aggregate_2b(x_2a, x_3a)
        x_1c = self.aggregate_1c(x_1b, x_2b)
        return self.conv_1x1(x_1c)

class LaserNet_LiDAR(nn.Module):
    def __init__(self, deep_aggregation_num_channels=[32, 64, 128], num_out_channels=9, lidar_in_channels=5):
        super().__init__()
        self.Lidar_CNN = nn.Conv2d(lidar_in_channels, 64, kernel_size=3, padding=1)
        self.DL = Deep_Aggregation(64, deep_aggregation_num_channels, num_out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        pi = 0.01
        bias_value = -np.log((1 - pi) / pi)
    
        final_layer = self.DL.conv_1x1
        nn.init.constant_(final_layer.bias, 0)
        
        final_layer.bias.data[0] = bias_value
        

    def forward(self, lidar):
        lidar_semantics = self.Lidar_CNN(lidar)
        out = self.DL(lidar_semantics)
        return out
