import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.nn_utils import ConvBlock, conv2, conv4, deconv, deconv3, warp_fn

class DownsampleImage(nn.Module):
    def __init__(self, num_channels, padding_mode='zeros'):
        super().__init__()
        self.downsample_mask = nn.Sequential(
            ConvBlock(num_channels, 2*num_channels, kernel_size=2, padding_mode=padding_mode),
            ConvBlock(2*num_channels, 2*num_channels, kernel_size=5, padding_mode=padding_mode),
            ConvBlock(2*num_channels, 2*num_channels, kernel_size=3, padding_mode=padding_mode),
            ConvBlock(2*num_channels, 25, kernel_size=1, padding_mode=padding_mode, act=None)
        )

    def forward(self, x, img):
        """ down-sample the image [H*2, W*2, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = img.shape      
        
        mask = self.downsample_mask(x)
        mask = mask.view(N, 1, 25, H // 2, W // 2)
        mask = torch.softmax(mask, dim=2)  
        down_img = F.unfold(img, [5,5], stride=2, padding=2)
        down_img = down_img.view(N, 1, 25, H // 2, W // 2) 
        down_img = torch.sum(mask * down_img, dim=2)
        return down_img

class ContextNet(nn.Module):
    def __init__(self, num_channels, padding_mode='zeros'):
        super().__init__()

        chs = [1, 1*num_channels, 2*num_channels, 4*num_channels, 8*num_channels]

        self.convs = nn.ModuleList([conv2(ch_in, ch_out, padding_mode=padding_mode) for ch_in, ch_out in zip(chs[:-1], chs[1:])])

    def forward(self, feat, flow):
        feat_pyramid = []

        for conv in self.convs:
            feat = conv(feat)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
            warped_feat = warp_fn(feat, flow)
            feat_pyramid.append(warped_feat)

        return feat_pyramid

class RefineUNet(nn.Module):
    def __init__(self, num_channels, padding_mode='zeros'):
        super().__init__()
        
        self.down1 = conv4(8, 2*num_channels, padding_mode=padding_mode)
        self.down2 = conv2(4*num_channels, 4*num_channels, padding_mode=padding_mode)
        self.down3 = conv2(8*num_channels, 8*num_channels, padding_mode=padding_mode)
        self.down4 = conv2(16*num_channels, 16*num_channels, padding_mode=padding_mode)
        self.up1 = deconv(32*num_channels, 8*num_channels, padding_mode=padding_mode)
        self.up2 = deconv(16*num_channels, 4*num_channels, padding_mode=padding_mode)
        self.up3 = deconv(8*num_channels, 2*num_channels, padding_mode=padding_mode)
        self.up4 = deconv3(4*num_channels, num_channels, padding_mode=padding_mode)

    def forward(self, cat, c0, c1):
        s0 = self.down1(cat)
        s1 = self.down2(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down3(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down4(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up1(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up2(torch.cat((x, s2), 1))
        x = self.up3(torch.cat((x, s1), 1))
        x = self.up4(torch.cat((x, s0), 1))
        return x

class FusionNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.padding_mode = args.padding_mode
        
        self.contextnet = ContextNet(args.num_channels, padding_mode=self.padding_mode)
        self.unet = RefineUNet(args.num_channels, padding_mode=self.padding_mode)
        self.refine_pred = ConvBlock(args.num_channels, 2, kernel_size=3, padding_mode=self.padding_mode)

        self.downsample_image = DownsampleImage(args.num_channels, padding_mode=self.padding_mode)

        # fix the parameters if needed
        if ("fix_params" in args) and (args.fix_params):
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, img0, img1, bi_flow):
        # upsample input images and estimated bi_flow
        img0 = F.interpolate(input=img0, scale_factor=2, mode="bilinear", align_corners=False)
        img1 = F.interpolate(input=img1, scale_factor=2, mode="bilinear", align_corners=False)
        bi_flow = F.interpolate(input=bi_flow, scale_factor=2, mode="bilinear", align_corners=False) * 2

        # input features for sythesis network: original images, warped images, warped features, and flow_0t_1t
        flow_0t = bi_flow[:, :2] * 0.5
        flow_1t = bi_flow[:, 2:4] * 0.5
        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
        
        warped_img0 = warp_fn(img0, flow_0t)
        warped_img1 = warp_fn(img1, flow_1t)
        c0 = self.contextnet(img0, flow_0t)
        c1 = self.contextnet(img1, flow_1t)

        # feature extraction by u-net
        x = self.unet(torch.cat((warped_img0, warped_img1, img0, img1, flow_0t_1t), 1), c0, c1)

        # prediction
        refine = torch.sigmoid(self.refine_pred(x))
        refine_res = refine[:, 0:1] * 2 - 1
        refine_mask = refine[:, 1:2]
        merged_img = warped_img0 * refine_mask + warped_img1 * (1 - refine_mask)
        interp_img = merged_img + refine_res

        # convex down-sampling
        interp_img = self.downsample_image(x, interp_img)

        return interp_img

if __name__ == "__main__":
    pass
