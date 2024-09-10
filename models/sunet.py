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
        self.depth = depth
        n1 = base_channel
        filters = [n1 * (2**i) for i in range(depth)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ModuleList([Up(filters[i], bilinear) for i in range(1, depth)])

        # Encoder (downsampling)
        self.conv_blocks_down = nn.ModuleList(
            [
                ConvBlockNested(
                    in_channels if i == 0 else filters[i - 1], filters[i], filters[i]
                )
                for i in range(depth)
            ]
        )

        # Decoder (upsampling)
        self.conv_blocks_up = nn.ModuleDict()
        for i in range(depth - 1):
            for j in range(depth - 1 - i):
                in_channels = filters[j] * (i + 2) + filters[j + 1]
                self.conv_blocks_up[f"{i}_{j}"] = ConvBlockNested(
                    in_channels, filters[j], filters[j]
                )

        self.ca = ChannelAttention(filters[0] * (depth - 1), ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(
            filters[0] * (depth - 1), num_classes, kernel_size=1
        )

    def forward(self, xA, xB):
        encodersA = [self.conv_blocks_down[0](xA)]
        encodersB = [self.conv_blocks_down[0](xB)]

        # Encoder
        for i in range(1, self.depth):
            encodersA.append(self.conv_blocks_down[i](self.pool(encodersA[-1])))
            encodersB.append(self.conv_blocks_down[i](self.pool(encodersB[-1])))

        # Decoder
        decoders = []
        for i in range(self.depth - 1):
            x = []
            for j in range(self.depth - 1 - i):
                if i == 0:
                    inp = torch.cat(
                        [encodersA[j], encodersB[j], self.up[j](encodersB[j + 1])], 1
                    )
                else:
                    inp = torch.cat(
                        [
                            encodersA[j],
                            encodersB[j],
                            decoders[i - 1][j],
                            self.up[j](decoders[i - 1][j + 1]),
                        ],
                        1,
                    )
                x.append(self.conv_blocks_up[f"{i}_{j}"](inp))
            decoders.append(x)

        # Final layer
        out = torch.cat(decoders[-1], 1)
        intra = torch.sum(torch.stack(decoders[-1]), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, self.depth - 1, 1, 1))
        out = self.conv_final(out)

        return out
