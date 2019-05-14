import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_


DEBUG = False
__all__ = ['flownetl']


def conv_dw(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=1, groups=in_planes, bias=False),
        nn.BatchNorm2d(in_planes),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),

        nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


class FlowNetL(nn.Module):
    expansion = 1

    def __init__(self):
        super(FlowNetL, self).__init__()

        # Encoder
        self.conv1 = conv_dw(in_planes=6, out_planes=32, stride=2)
        self.conv2 = conv_dw(in_planes=32, out_planes=64, stride=2)
        self.conv3 = conv_dw(in_planes=64, out_planes=128, stride=2)
        self.conv4a = conv_dw(in_planes=128, out_planes=256, stride=2)
        self.conv4b = conv_dw(in_planes=256, out_planes=256, stride=1)
        self.conv5a = conv_dw(in_planes=256, out_planes=512, stride=2)
        self.conv5b = conv_dw(in_planes=512, out_planes=512, stride=1)
        self.conv6a = conv_dw(in_planes=512, out_planes=1024, stride=2)
        self.conv6b = conv_dw(in_planes=1024, out_planes=1024, stride=1)

        # Decoder
        self.conv7 = conv_dw(in_planes=1024, out_planes=256, stride=1)
        self.conv8 = conv_dw(in_planes=768, out_planes=128, stride=1)
        self.conv9 = conv_dw(in_planes=384, out_planes=64, stride=1)
        self.conv10 = conv_dw(in_planes=192, out_planes=32, stride=1)
        self.conv11 = conv_dw(in_planes=96, out_planes=16, stride=1)

        # Optical Flow Predictors
        self.conv12 = conv_dw(in_planes=256, out_planes=2, stride=1)
        self.conv13 = conv_dw(in_planes=128, out_planes=2, stride=1)
        self.conv14 = conv_dw(in_planes=64, out_planes=2, stride=1)
        self.conv15 = conv_dw(in_planes=32, out_planes=2, stride=1)
        self.conv16 = conv_dw(in_planes=16, out_planes=2, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        # Encoder
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4a = self.conv4a(out_conv3)
        out_conv4b = self.conv4b(out_conv4a)
        out_conv5a = self.conv5a(out_conv4b)
        out_conv5b = self.conv5b(out_conv5a)
        out_conv6a = self.conv6a(out_conv5b)
        out_conv6b = self.conv6b(out_conv6a)

        # Decoder
        out_conv7 = self.conv7(out_conv6b)
        concat7_5b = torch.cat((F.interpolate(input=out_conv7,
                                              scale_factor=2,
                                              mode="nearest"),
                                out_conv5b),
                               dim=1)
        out_conv8 = self.conv8(concat7_5b)
        concat8_4b = torch.cat((F.interpolate(input=out_conv8,
                                              scale_factor=2,
                                              mode="nearest"),
                                out_conv4b),
                               dim=1)
        out_conv9 = self.conv9(concat8_4b)
        concat9_3 = torch.cat((F.interpolate(input=out_conv9,
                                             scale_factor=2,
                                             mode="nearest"),
                               out_conv3),
                              dim=1)
        out_conv10 = self.conv10(concat9_3)
        concat10_2 = torch.cat((F.interpolate(input=out_conv10,
                                              scale_factor=2,
                                              mode="nearest"),
                                out_conv2),
                               dim=1)
        out_conv11 = self.conv11(concat10_2)

        # Optical Flow Predictiors
        out_conv12 = self.conv12(out_conv7)
        out_conv13 = self.conv13(out_conv8)
        out_conv14 = self.conv14(out_conv9)
        out_conv15 = self.conv15(out_conv10)
        out_conv16 = self.conv16(out_conv11)

        # Multiple Optical Flow Predictions Fusion
        upsample_16x_out_conv12 = F.interpolate(input=out_conv12,
                                                scale_factor=16,
                                                mode="nearest")
        upsample_8x_out_conv13 = F.interpolate(input=out_conv13,
                                               scale_factor=8,
                                               mode="nearest")
        upsample_4x_out_conv14 = F.interpolate(input=out_conv14,
                                               scale_factor=4,
                                               mode="nearest")
        upsample_2x_out_conv15 = F.interpolate(input=out_conv15,
                                               scale_factor=2,
                                               mode="nearest")

        avg_out = (upsample_16x_out_conv12 + upsample_8x_out_conv13 +
                   upsample_4x_out_conv14 + upsample_2x_out_conv15 +
                   out_conv16) / 5.

        if DEBUG:
            print("=============== FlowNetL Shape ===============")
            print("Input:")
            print(f"input.shape: {x.shape}")
            print()
            print("Encoder: ")
            print(f"out_conv1.shape: {out_conv1.shape}")
            print(f"out_conv2.shape: {out_conv2.shape}")
            print(f"out_conv3.shape: {out_conv3.shape}")
            print(f"out_conv4a.shape: {out_conv4a.shape}")
            print(f"out_conv4b.shape: {out_conv4b.shape}")
            print(f"out_conv5a.shape: {out_conv5a.shape}")
            print(f"out_conv5b.shape: {out_conv5b.shape}")
            print(f"out_conv6a.shape: {out_conv6a.shape}")
            print(f"out_conv6b.shape: {out_conv6b.shape}")
            print()
            print("Decoder: ")
            print(f"out_conv7.shape: {out_conv7.shape}")
            print(f"concat7_5b.shape: {concat7_5b.shape}")
            print(f"out_conv8.shape: {out_conv8.shape}")
            print(f"concat8_4b.shape: {concat8_4b.shape}")
            print(f"out_conv9.shape: {out_conv9.shape}")
            print(f"concat9_3.shape: {concat9_3.shape}")
            print(f"out_conv10.shape: {out_conv10.shape}")
            print(f"concat10_2.shape: {concat10_2.shape}")
            print(f"out_conv11.shape: {out_conv11.shape}")
            print()
            print("Optical Flow Predictors: ")
            print(f"out_conv12.shape: {out_conv12.shape}")
            print(f"out_conv13.shape: {out_conv13.shape}")
            print(f"out_conv14.shape: {out_conv14.shape}")
            print(f"out_conv15.shape: {out_conv15.shape}")
            print(f"out_conv16.shape: {out_conv16.shape}")
            print()
            print("Multiple Optical Flow Predictions Fusion:")
            print(f"upsample_16x_out_conv12.shape: {upsample_16x_out_conv12.shape}")
            print(f"upsample_8x_out_conv13.shape: {upsample_8x_out_conv13.shape}")
            print(f"upsample_4x_out_conv14.shape: {upsample_4x_out_conv14.shape}")
            print(f"upsample_2x_out_conv15.shape: {upsample_2x_out_conv15.shape}")
            print(f"avg_out.shape: {avg_out.shape}")
            print()

        return avg_out

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownetl(weights=None):
    """
    Args:
        weights : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetL()
    if weights is not None:
        model.load_state_dict(weights['state_dict'])
    return model


if __name__ == '__main__':
    B, C, H, W = 2, 6, 384, 512
    image = torch.randn(B, C, H, W, requires_grad=True)
    model = flownetl()
    flow = model(image)
