import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.feature import BasicBlock, BasicConv, Conv2x
from nets.warp import disp_warp


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class NormalizationRefinement(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(NormalizationRefinement, self).__init__()

        # Left and warped error
        in_channels = 6
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.conv_start = BasicConv(32, 32, kernel_size=3, padding=2, dilation=2)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):

        assert low_disp.dim() == 4
        low_disp = -low_disp
        # low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # min_disp = torch.min(disp)
        # max_disp = torch.max(disp)

        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        disp = disp / disp.shape[3] * 1024

        # Warp right image to left view with current disparity
        error = warped_right - left_img  # [B, C, H, W]
        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]
        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        residual_disp = self.final_conv(x)  # [B, 1, H, W]

        new_disp = F.leaky_relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]

        disp = new_disp / 1024 * disp.shape[3]  # dtu_test_tank

        return -disp
