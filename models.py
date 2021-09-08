import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from modules import ConvBlock2D, DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBlock3D, ResBottleneck
from utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)


class AFE(nn.Module):
    # 3D appearance features extractor
    # [N,3,256,256]
    # [N,64,256,256]
    # [N,128,128,128]
    # [N,256,64,64]
    # [N,512,64,64]
    # [N,32,16,64,64]
    def __init__(self, use_weight_norm=False, down_seq=[64, 128, 256], n_res=6, C=32, D=16):
        super().__init__()
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], C * D, 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock3D(C, use_weight_norm) for _ in range(n_res)])
        self.C, self.D = C, D

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.res(x)
        return x


class CKD(nn.Module):
    # Canonical keypoints detector
    # [N,3,256,256]
    # [N,64,128,128]
    # [N,128,64,64]
    # [N,256,32,32]
    # [N,512,16,16]
    # [N,1024,8,8]
    # [N,16384,8,8]
    # [N,1024,16,8,8]
    # [N,512,16,16,16]
    # [N,256,16,32,32]
    # [N,128,16,64,64]
    # [N,64,16,128,128]
    # [N,32,16,256,256]
    # [N,20,16,256,256] (heatmap)
    # [N,20,3] (key points)
    def __init__(
        self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], up_seq=[1024, 512, 256, 128, 64, 32], D=16, K=15, scale_factor=0.25
    ):
        super().__init__()
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.up(x)
        x = self.out_conv(x)
        heatmap = out2heatmap(x)
        kp = heatmap2kp(heatmap)
        return kp


class HPE_EDE(nn.Module):
    # Head pose estimator && expression deformation estimator
    # [N,3,256,256]
    # [N,64,64,64]
    # [N,256,64,64]
    # [N,512,32,32]
    # [N,1024,16,16]
    # [N,2048,8,8]
    # [N,2048]
    # [N,66] [N,66] [N,66] [N,3] [N,60]
    # [N,] [N,] [N,] [N,3] [N,20,3]
    def __init__(self, use_weight_norm=False, n_filters=[64, 256, 512, 1024, 2048], n_blocks=[3, 3, 5, 2], n_bins=66, K=15):
        super().__init__()
        self.pre_layers = nn.Sequential(ConvBlock2D("CNA", 3, n_filters[0], 7, 2, 3, use_weight_norm), nn.MaxPool2d(3, 2, 1))
        res_layers = []
        for i in range(len(n_filters) - 1):
            res_layers.extend(self._make_layer(i, n_filters[i], n_filters[i + 1], n_blocks[i], use_weight_norm))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc_yaw = nn.Linear(n_filters[-1], n_bins)
        self.fc_pitch = nn.Linear(n_filters[-1], n_bins)
        self.fc_roll = nn.Linear(n_filters[-1], n_bins)
        self.fc_t = nn.Linear(n_filters[-1], 3)
        self.fc_delta = nn.Linear(n_filters[-1], 3 * K)
        self.n_bins = n_bins
        self.idx_tensor = torch.FloatTensor(list(range(self.n_bins))).unsqueeze(0).cuda()

    def _make_layer(self, i, in_channels, out_channels, n_block, use_weight_norm):
        stride = 1 if i == 0 else 2
        return [ResBottleneck(in_channels, out_channels, stride, use_weight_norm)] + [
            ResBottleneck(out_channels, out_channels, 1, use_weight_norm) for _ in range(n_block)
        ]

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = torch.mean(x, (2, 3))
        yaw, pitch, roll, t, delta = self.fc_yaw(x), self.fc_pitch(x), self.fc_roll(x), self.fc_t(x), self.fc_delta(x)
        yaw = torch.softmax(yaw, dim=1)
        pitch = torch.softmax(pitch, dim=1)
        roll = torch.softmax(roll, dim=1)
        yaw = (yaw * self.idx_tensor).sum(dim=1)
        pitch = (pitch * self.idx_tensor).sum(dim=1)
        roll = (roll * self.idx_tensor).sum(dim=1)
        yaw = (yaw - self.n_bins // 2) * 3 * np.pi / 180
        pitch = (pitch - self.n_bins // 2) * 3 * np.pi / 180
        roll = (roll - self.n_bins // 2) * 3 * np.pi / 180
        delta = delta.view(x.shape[0], -1, 3)
        return yaw, pitch, roll, t, delta


class MFE(nn.Module):
    # Motion field estimator
    # (4+1)x(20+1)=105
    # [N,105,16,64,64]
    # ...
    # [N,32,16,64,64]
    # [N,137,16,64,64]
    # 1.
    # [N,21,16,64,64] (mask)
    # 2.
    # [N,2192,64,64]
    # [N,1,64,64] (occlusion)
    def __init__(self, use_weight_norm=False, down_seq=[80, 64, 128, 256, 512, 1024], up_seq=[1024, 512, 256, 128, 64, 32], K=15, D=16, C1=32, C2=4):
        super().__init__()
        self.compress = nn.Conv3d(C1, C2, 1, 1, 0)
        self.down = nn.Sequential(*[DownBlock3D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.mask_conv = nn.Conv3d(down_seq[0] + up_seq[-1], K + 1, 7, 1, 3)
        self.occlusion_conv = nn.Conv2d((down_seq[0] + up_seq[-1]) * D, 1, 7, 1, 3)
        self.C, self.D = down_seq[0] + up_seq[-1], D

    def forward(self, fs, kp_s, kp_d, Rs, Rd):
        # the original fs is compressed to 4 channels using a 1x1x1 conv
        fs_compressed = self.compress(fs)
        N, _, D, H, W = fs.shape
        # [N,21,1,16,64,64]
        heatmap_representation = create_heatmap_representations(fs_compressed, kp_s, kp_d)
        # [N,21,16,64,64,3]
        sparse_motion = create_sparse_motions(fs_compressed, kp_s, kp_d, Rs, Rd)
        # [N,21,4,16,64,64]
        deformed_source = create_deformed_source_image(fs_compressed, sparse_motion)
        input = torch.cat([heatmap_representation, deformed_source], dim=2).view(N, -1, D, H, W)
        output = self.down(input)
        output = self.up(output)
        x = torch.cat([input, output], dim=1)
        mask = self.mask_conv(x)
        # [N,21,16,64,64,1]
        mask = F.softmax(mask, dim=1).unsqueeze(-1)
        # [N,16,64,64,3]
        deformation = (sparse_motion * mask).sum(dim=1)
        occlusion = self.occlusion_conv(x.view(N, -1, H, W))
        occlusion = torch.sigmoid(occlusion)
        return deformation, occlusion


class Generator(nn.Module):
    # Generator
    # [N,32,16,64,64]
    # [N,512,64,64]
    # [N,256,64,64]
    # [N,128,128,128]
    # [N,64,256,256]
    # [N,3,256,256]
    def __init__(self, use_weight_norm=True, n_res=6, up_seq=[256, 128, 64], D=16, C=32):
        super().__init__()
        self.in_conv = ConvBlock2D("CNA", C * D, up_seq[0], 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.mid_conv = nn.Conv2d(up_seq[0], up_seq[0], 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock2D(up_seq[0], use_weight_norm) for _ in range(n_res)])
        self.up = nn.Sequential(*[UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv2d(up_seq[-1], 3, 7, 1, 3)

    def forward(self, fs, deformation, occlusion):
        N, _, D, H, W = fs.shape
        fs = F.grid_sample(fs, deformation, align_corners=True).view(N, -1, H, W)
        fs = self.in_conv(fs)
        fs = self.mid_conv(fs)
        fs = fs * occlusion
        fs = self.res(fs)
        fs = self.up(fs)
        fs = self.out_conv(fs)
        return fs


class Discriminator(nn.Module):
    # Patch Discriminator

    def __init__(self, use_weight_norm=True, down_seq=[64, 128, 256, 512], K=15):
        super().__init__()
        layers = []
        layers.append(ConvBlock2D("CNA", 3 + K, down_seq[0], 3, 2, 1, use_weight_norm, "instance", "leakyrelu"))
        layers.extend(
            [
                ConvBlock2D("CNA", down_seq[i], down_seq[i + 1], 3, 2 if i < len(down_seq) - 2 else 1, 1, use_weight_norm, "instance", "leakyrelu")
                for i in range(len(down_seq) - 1)
            ]
        )
        layers.append(ConvBlock2D("CN", down_seq[-1], 1, 3, 1, 1, use_weight_norm, activation_type="none"))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, kp):
        heatmap = kp2gaussian_2d(kp.detach()[:, :, :2], x.shape[2:])
        x = torch.cat([x, heatmap], dim=1)
        res = [x]
        for layer in self.layers:
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
