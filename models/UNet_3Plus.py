# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.init_weights import init_weights
import numpy as np


"""
    UNet 3+ with deep supervision and class-guided module
"""


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)

        return x


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


class UNet_3Plus_DeepSup_CGM(nn.Module):
    def __init__(
        self,
        in_channels=12,
        n_classes=1,
        feature_scale=4,
        base_channel=64,
        is_deconv=True,
        is_batchnorm=True,
    ):
        super(UNet_3Plus_DeepSup_CGM, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.base_channel = base_channel

        filters = [
            base_channel,
            base_channel * 2,
            base_channel * 4,
            base_channel * 8,
            base_channel * 16,
        ]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.concat1 = unetConv2(filters[0] * 2, filters[0], self.is_batchnorm)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.concat2 = unetConv2(filters[1] * 2, filters[1], self.is_batchnorm)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.concat3 = unetConv2(filters[2] * 2, filters[2], self.is_batchnorm)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.concat4 = unetConv2(filters[3] * 2, filters[3], self.is_batchnorm)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.concat5 = unetConv2(filters[4] * 2, filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        """stage 4d"""
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        """stage 3d"""
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode="bilinear")  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        """stage 2d """
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="bilinear")  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="bilinear")  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        """stage 1d"""
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="bilinear")  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="bilinear")  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="bilinear")  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode="bilinear")  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.upscore4 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.upscore3 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(filters[4], 2, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid(),
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, xA, xB):
        ## -------------Encoder-------------
        h1_0 = self.conv1(xA)  # h1->320*320*64
        h1_1 = self.conv1(xB)

        h2_0 = self.maxpool1(h1_0)
        h2_1 = self.maxpool1(h1_1)
        h2_0 = self.conv2(h2_0)  # h2->160*160*128
        h2_1 = self.conv2(h2_1)

        h3_0 = self.maxpool2(h2_0)
        h3_1 = self.maxpool2(h2_1)
        h3_0 = self.conv3(h3_0)  # h3->80*80*256
        h3_1 = self.conv3(h3_1)

        h4_0 = self.maxpool3(h3_0)
        h4_1 = self.maxpool3(h3_1)
        h4_0 = self.conv4(h4_0)  # h4->40*40*512
        h4_1 = self.conv4(h4_1)

        h5_0 = self.maxpool4(h4_0)
        h5_1 = self.maxpool4(h4_1)
        hd5_0 = self.conv5(h5_0)  # h5->20*20*1024
        hd5_1 = self.conv5(h5_1)

        ## -------------Concat-------------

        h1 = self.concat1(torch.cat([h1_0, h1_1], 1))
        h2 = self.concat2(torch.cat([h2_0, h2_1], 1))
        h3 = self.concat3(torch.cat([h3_0, h3_1], 1))
        h4 = self.concat4(torch.cat([h4_0, h4_1], 1))
        hd5 = self.concat5(torch.cat([hd5_0, hd5_1], 1))

        # -------------Classification-------------
        cls_branch = self.cls(hd5).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(
            self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        )
        h2_PT_hd4 = self.h2_PT_hd4_relu(
            self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        )
        h3_PT_hd4 = self.h3_PT_hd4_relu(
            self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        )
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(
            self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        )
        hd4 = self.relu4d_1(
            self.bn4d_1(
                self.conv4d_1(
                    torch.cat(
                        (h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1
                    )
                )
            )
        )  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(
            self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        )
        h2_PT_hd3 = self.h2_PT_hd3_relu(
            self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        )
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(
            self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        )
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(
            self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        )
        hd3 = self.relu3d_1(
            self.bn3d_1(
                self.conv3d_1(
                    torch.cat(
                        (h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1
                    )
                )
            )
        )  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(
            self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        )
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(
            self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        )
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(
            self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        )
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(
            self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        )
        hd2 = self.relu2d_1(
            self.bn2d_1(
                self.conv2d_1(
                    torch.cat(
                        (h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1
                    )
                )
            )
        )  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(
            self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        )
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(
            self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        )
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(
            self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        )
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(
            self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        )
        hd1 = self.relu1d_1(
            self.bn1d_1(
                self.conv1d_1(
                    torch.cat(
                        (h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1
                    )
                )
            )
        )  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        d5 = self.dotProduct(d5, cls_branch_max)

        return d1, d2, d3, d4, d5
