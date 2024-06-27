import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 2  # Reduced expansion
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x

class SimpleConformer(nn.Module):

    def __init__(self, patch_size=8, in_chans=3, num_classes=10, base_channel=32, channel_ratio=2, num_med_block=0,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        super().__init__()
        self.num_classes = num_classes
        stage_1_channel = int(base_channel * channel_ratio)

        # Stem stage: get the feature maps by conv block
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 1 / 4 [56, 56]

        # 1 stage
        self.conv_1 = ConvBlock(inplanes=32, outplanes=stage_1_channel, res_conv=True, stride=1)

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 2~3 stage
        for i in range(2, 4):
            s = 2 if i == 2 else 1
            in_channel = stage_1_channel if i == 2 else stage_2_channel
            res_conv = True if i == 2 else False
            self.add_module('conv_block_' + str(i),
                ConvBlock(
                    inplanes=in_channel, outplanes=stage_2_channel, res_conv=res_conv, stride=s))

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 4 stage
        s = 2
        in_channel = stage_2_channel
        res_conv = True
        self.add_module('conv_block_4',
            ConvBlock(
                inplanes=in_channel, outplanes=stage_3_channel, res_conv=res_conv, stride=s))

        self.fin_stage = 5
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(stage_3_channel, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem stage
        x = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        x, _ = self.conv_1(x)

        # Conv blocks
        for i in range(2, self.fin_stage):
            x, _ = eval('self.conv_block_' + str(i))(x)

        # Classification
        x = self.pooling(x).flatten(1)
        x = self.conv_cls_head(x)

        return x


num_classes = 4  # Adjust this based on your dataset
model = SimpleConformer(patch_size=8, in_chans=3, num_classes=num_classes, base_channel=32, channel_ratio=2, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)