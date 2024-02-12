"""
S. Fang, K. Li, J. Shao, and Z. Li, 
“SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images,” 
IEEE Geosci. Remote Sensing Lett., pp. 1-5, 2021, doi: 10.1109/LGRS.2021.3056416.
"""

import torch
import torch.nn as nn


class ConvBlockNested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlockNested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)) + x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SNUNet_ECAM(nn.Module):
    def __init__(
        self, in_channels, num_classes, base_channel=32, depth=5, bilinear=False
    ):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        filters = [base_channel * (2**i) for i in range(depth)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layers using lists
        self.conv_layers = [
            ConvBlockNested(
                in_channels if i == 0 else filters[i - 1], filters[i], filters[i]
            )
            for i in range(depth)
        ]

        # Upsampling layers using lists
        self.up_layers = [Up(filters[i]) for i in range(1, depth)]

        # Remaining unchanged components
        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
        self.conv_final = nn.Conv2d(filters[0] * 4, num_classes, kernel_size=1)

    def forward(self, xA, xB):
        # Forward pass for both inputs
        saved_features_A = []
        saved_features_B = []
        current_A = xA
        current_B = xB

        # Pass inputs through the convolutional layers, saving intermediate features
        for conv in self.conv_layers:
            current_A = conv(current_A)
            current_B = conv(current_B)
            saved_features_A.append(current_A)
            saved_features_B.append(current_B)

        # Combine and upsample features
        output_features = []
        for i in range(len(self.up_layers)):
            # Upsample and concatenate features from both streams and previous layers
            upsampled_A = self.up_layers[i](saved_features_A[-(i + 2)])
            upsampled_B = self.up_layers[i](saved_features_B[-(i + 2)])
            combined = torch.cat(
                [
                    saved_features_A[-(i + 1)],
                    saved_features_B[-(i + 1)],
                    upsampled_A,
                    upsampled_B,
                ],
                1,
            )
            output_features.append(combined)

        # Concatenate all output features
        out = torch.cat(output_features, 1)

        # Apply channel attention and final convolution
        intra = torch.sum(torch.stack(output_features), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        return out
