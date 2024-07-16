import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.nn_utils import ConvBlock, backwarp, conv2, deconv, warp_fn


class ImportanceMask(nn.Module):
    def __init__(self, mode):
        super().__init__()

        self.conv_img = ConvBlock(
            in_channels=1, out_channels=4, kernel_size=3, padding_mode=mode, act=None
        )
        self.conv_error = ConvBlock(
            in_channels=1, out_channels=4, kernel_size=3, padding_mode=mode, act=None
        )

        self.conv_mix = MaskUNet(chs_in=8, num_channels=4, padding_mode=mode)

    def forward(self, img, error):
        img = self.conv_img(img)
        error = self.conv_error(error)

        feat = torch.cat((img, error), dim=1)

        feat = self.conv_mix(feat)

        return feat


class MaskUNet(nn.Module):
    def __init__(self, chs_in, num_channels, padding_mode="zeros"):
        super().__init__()

        self.down1 = conv2(chs_in, 2 * num_channels, padding_mode=padding_mode)
        self.down2 = conv2(
            2 * num_channels, 4 * num_channels, padding_mode=padding_mode
        )
        self.down3 = conv2(
            4 * num_channels, 8 * num_channels, padding_mode=padding_mode
        )
        self.down4 = conv2(
            8 * num_channels, 16 * num_channels, padding_mode=padding_mode
        )
        self.up1 = deconv(
            16 * num_channels, 8 * num_channels, padding_mode=padding_mode
        )
        self.up2 = deconv(
            16 * num_channels, 4 * num_channels, padding_mode=padding_mode
        )
        self.up3 = deconv(8 * num_channels, 2 * num_channels, padding_mode=padding_mode)
        self.up4 = deconv(4 * num_channels, 1, padding_mode=padding_mode)

    def forward(self, s0):
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.up1(x)
        x = self.up2(torch.cat((x, s3), 1))
        x = self.up3(torch.cat((x, s2), 1))
        x = self.up4(torch.cat((x, s1), 1))
        return x


# **************************************************************************************************#
# => Feature Pyramid
# **************************************************************************************************#
class FeatPyramid(nn.Module):
    """Two-level feature pyramid
    1) remove high-level feature pyramid (compared to PWC-Net), and add more conv layers to stage 2;
    2) do not increase the output channel of stage 2, in order to keep the cost of corr volume under control.
    """

    def __init__(self, num_channels=24, kernel_size=3, mode="zeros"):
        super().__init__()
        act = "lrelu"
        self.conv_stage1 = nn.Sequential(
            ConvBlock(1, num_channels, kernel_size=2, padding_mode=mode, act=act),
            *[
                ConvBlock(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    padding_mode=mode,
                    act=act,
                )
                for _ in range(2)
            ]
        )
        self.conv_stage2 = nn.Sequential(
            ConvBlock(
                num_channels,
                2 * num_channels,
                kernel_size=2,
                padding_mode=mode,
                act=act,
            ),
            *[
                ConvBlock(
                    2 * num_channels,
                    2 * num_channels,
                    kernel_size=kernel_size,
                    padding_mode=mode,
                    act=act,
                )
                for _ in range(5)
            ]
        )

    def forward(self, img):
        x = self.conv_stage1(img)
        x = self.conv_stage2(x)

        return x


# **************************************************************************************************#
# => Estimator
# **************************************************************************************************#


class Correlation(nn.Module):
    def __init__(self, corr_radius=4):
        super().__init__()
        self.corr_radius = [corr_radius] * 4

    def forward(self, x1, x2):
        B, C, H, W = x1.size()

        x2 = F.pad(x2, self.corr_radius)

        # Using unfold function to create patches for correlation
        kv = x2.unfold(2, H, 1).unfold(3, W, 1)

        # We need to consider the correlation in the channel dimension. Therefore, reshape kv accordingly.
        kv = kv.contiguous().view(B, C, -1, H, W)

        # Calculating correlation
        cv = (x1.view(B, C, 1, H, W) * kv).mean(dim=1, keepdim=True)

        # Reshaping the output to match the shape of the output of the original function
        cv = cv.view(B, -1, H, W)

        return cv


class Estimator(nn.Module):
    """A 6-layer flow estimator, with correlation-injected features
    1) construct partial cost volume with the CNN features from stage 2 of the feature pyramid;
    2) estimate bi-directional flows, by feeding cost volume, CNN features for both warped images,
    CNN feature and estimated flow from previous iteration.
    """

    def __init__(self, pyr_dim=24, kernel_size=3, corr_radius=4, mode="zeros"):
        super().__init__()
        image_feat_channel = 2 * pyr_dim
        last_flow_feat_channel = 64
        in_channels = (
            (corr_radius * 2 + 1) ** 2
            + image_feat_channel * 2
            + last_flow_feat_channel
            + 4
        )

        self.corr = Correlation(corr_radius)

        act = "lrelu"
        self.convs = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=160,
                kernel_size=1,
                padding_mode=mode,
                act=act,
            ),
            ConvBlock(
                in_channels=160,
                out_channels=128,
                kernel_size=kernel_size,
                padding_mode=mode,
                act=act,
            ),
            ConvBlock(
                in_channels=128,
                out_channels=112,
                kernel_size=kernel_size,
                padding_mode=mode,
                act=act,
            ),
            ConvBlock(
                in_channels=112,
                out_channels=96,
                kernel_size=kernel_size,
                padding_mode=mode,
                act=act,
            ),
            ConvBlock(
                in_channels=96,
                out_channels=64,
                kernel_size=kernel_size,
                padding_mode=mode,
                act=act,
            ),
        )

        self.final_conv = ConvBlock(
            in_channels=64,
            out_channels=4,
            kernel_size=kernel_size,
            padding_mode=mode,
            act=None,
        )

    def forward(self, feat0, feat1, last_feat, last_flow):
        volume = F.leaky_relu(
            input=self.corr(feat0, feat1), negative_slope=0.1, inplace=False
        )

        feat = torch.cat([volume, feat0, feat1, last_feat, last_flow], 1)

        feat = self.convs(feat)
        flow = self.final_conv(feat)

        return flow, feat


class ForwardWarp(nn.Module):
    def __init__(self, warp_type, padding_mode):
        super().__init__()

        self.alpha = ImportanceMask(padding_mode) if warp_type == "soft_splat" else None

        if warp_type == "backwarp":
            self.fn = self.fn_backwarp
        elif warp_type == "avg_splat":
            self.fn = self.fn_avg_splat
        elif warp_type == "fw_splat":
            self.fn = self.fn_fw_splat
        elif warp_type == "soft_splat":
            self.fn = self.fn_soft_splat

    def fn_backwarp(self, img0, img1, flow):
        img0 = backwarp(tenInput=img0, tenFlow=flow[:, 2:])
        img1 = backwarp(tenInput=img1, tenFlow=flow[:, :2])
        return img0, img1

    def fn_avg_splat(self, img0, img1, flow):
        img0 = warp_fn(
            tenInput=img0, tenFlow=flow[:, :2] * 0.5, tenMetric=None, strType="average"
        )
        img1 = warp_fn(
            tenInput=img1, tenFlow=flow[:, 2:] * 0.5, tenMetric=None, strType="average"
        )
        return img0, img1

    def fn_fw_splat(self, img0, img1, flow):
        img0 = warp_fn(
            tenInput=img0, tenFlow=flow[:, :2] * 1.0, tenMetric=None, strType="average"
        )
        img1 = warp_fn(
            tenInput=img1, tenFlow=flow[:, 2:] * 1.0, tenMetric=None, strType="average"
        )
        return img0, img1

    def fn_soft_splat(self, img0, img1, flow):
        tenMetric0 = F.l1_loss(
            input=img0,
            target=backwarp(tenInput=img1, tenFlow=flow[:, :2] * 0.5),
            reduction="none",
        ).mean([1], True)
        tenMetric0 = self.alpha(img0, -tenMetric0).neg().clip(-20.0, 20.0)
        img0 = warp_fn(
            tenInput=img0,
            tenFlow=flow[:, :2] * 0.5,
            tenMetric=tenMetric0,
            strType="softmax",
        )

        tenMetric1 = F.l1_loss(
            input=img1,
            target=backwarp(tenInput=img0, tenFlow=flow[:, 2:] * 0.5),
            reduction="none",
        ).mean([1], True)
        tenMetric1 = self.alpha(img1, -tenMetric1).neg().clip(-20.0, 20.0)
        img1 = warp_fn(
            tenInput=img1,
            tenFlow=flow[:, 2:] * 0.5,
            tenMetric=tenMetric1,
            strType="softmax",
        )
        return img0, img1

    def forward(self, img0, img1, flow):
        return self.fn(img0, img1, flow)


# **************************************************************************************************#
# => BiFlowNet
# **************************************************************************************************#
class BiFlowNet(nn.Module):
    """Our bi-directional flownet
    In general, we combine image pyramid, middle-oriented forward warping,
    lightweight feature encoder and cost volume for simultaneous bi-directional
    motion estimation.
    """

    def __init__(self, args):
        super().__init__()
        self.last_flow_feat_channel = 64
        self.pyr_level = args.pyr_level
        self.warp_type = args.warp_type

        self.feat_pyramid = FeatPyramid(
            args.pyr_dim, args.kernel_size, args.padding_mode
        )
        self.flow_estimator = Estimator(
            args.pyr_dim, args.kernel_size, args.corr_radius, args.padding_mode
        )
        self.warp_imgs = ForwardWarp(args.warp_type, args.padding_mode)

        # fix the parameters if needed
        if ("fix_params" in args) and (args.fix_params):
            for p in self.parameters():
                p.requires_grad = False

    def pre_warp(self, img0, img1, last_flow):
        up_flow = (
            F.interpolate(
                input=last_flow, scale_factor=4.0, mode="bilinear", align_corners=False
            )
            * 4
        )
        img0, img1 = self.warp_imgs(img0, img1, up_flow)
        return img0, img1

    def forward_one_iteration(self, img0, img1, last_feat, last_flow):
        feat0 = self.feat_pyramid(img0)
        feat1 = self.feat_pyramid(img1)
        flow, feat = self.flow_estimator(feat0, feat1, last_feat, last_flow)
        return flow, feat

    def forward(self, img0, img1):
        N, _, H, W = img0.shape

        ###### First level
        level = self.pyr_level - 1
        scale_factor = 1 / 2**level
        img0_down = F.interpolate(
            input=img0, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )
        img1_down = F.interpolate(
            input=img1, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

        last_flow = torch.zeros(
            (N, 4, H // (2 ** (level + 2)), W // (2 ** (level + 2))), device=img0.device
        )
        last_feat = torch.zeros(
            (
                N,
                self.last_flow_feat_channel,
                H // (2 ** (level + 2)),
                W // (2 ** (level + 2)),
            ),
            device=img0.device,
        )

        flow, feat = self.forward_one_iteration(
            img0_down, img1_down, last_feat, last_flow
        )
        ######

        for level in list(range(self.pyr_level - 1))[::-1]:
            scale_factor = 1 / 2**level
            img0_down = F.interpolate(
                input=img0,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )
            img1_down = F.interpolate(
                input=img1,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )

            last_flow = (
                F.interpolate(
                    input=flow, scale_factor=2.0, mode="bilinear", align_corners=False
                )
                * 2
            )
            last_feat = F.interpolate(
                input=feat, scale_factor=2.0, mode="bilinear", align_corners=False
            )

            img0_down, img1_down = self.pre_warp(img0_down, img1_down, last_flow)
            flow, feat = self.forward_one_iteration(
                img0_down, img1_down, last_feat, last_flow
            )

        # directly up-sample estimated flow to full resolution with bi-linear interpolation
        output_flow = (
            F.interpolate(
                input=flow, scale_factor=4.0, mode="bilinear", align_corners=False
            )
            * 4
        )

        return output_flow


if __name__ == "__main__":
    pass
