import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from core.biflownet import BiFlowNet
from core.dataset import DatasetBase
from core.fusionnet import FusionNet


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        loss = torch.mean(torch.sqrt((x - y).pow(2) + self.eps))
        return loss


class TernaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 7
        self.out_channels = self.patch_size * self.patch_size
        self.padding = 1

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)  #########
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def hamming(self, t1, t2):
        dist = (t1 - t2).pow(2)
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t):
        n, _, h, w = t.size()
        inner = torch.ones(
            n, 1, h - 2 * self.padding, w - 2 * self.padding, device=t.device
        ).type_as(t)
        mask = F.pad(inner, [self.padding] * 4)
        return mask

    def forward(self, x, y):
        self.w = torch.eye(self.out_channels, device=x.device).reshape(
            (self.patch_size, self.patch_size, 1, self.out_channels)
        )
        self.w = self.w.permute(3, 2, 0, 1).float()
        x = self.transform(x)
        y = self.transform(y)
        return (self.hamming(x, y) * self.valid_mask(x)).mean()


class PhotometricLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ter_loss = TernaryLoss()
        self.char_loss = CharbonnierLoss()

    def forward(self, interp_img, gt):
        loss = 100 * (
            self.char_loss(interp_img, gt) + 0.1 * self.ter_loss(interp_img, gt)
        )
        return loss


def get_loss():
    return PhotometricLoss()


class CryoSamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.biflownet = BiFlowNet(cfg.biflownet)
        self.fusionnet = FusionNet(cfg.fusionnet)

        self.gap = cfg.train_data.max_frame_gap

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def validation(self, img0, img1):
        biflow = self.biflownet(img0, img1).contiguous()
        rec = self.fusionnet(img0, img1, biflow)
        rec = torch.clamp(rec, -1, 1)
        return rec

    def forward(self, img0, img1):
        biflow = self.biflownet(img0, img1).contiguous()
        rec = self.fusionnet(img0, img1, biflow)
        rec = torch.clamp(rec, -1, 1)
        return rec


def get_model(cfg, device, is_ddp, compile):
    model = CryoSamba(cfg).to(device=device)
    # if compile:
    #     model = torch.compile(model)
    model = DDP(model, device_ids=[device]) if is_ddp else model
    return model
