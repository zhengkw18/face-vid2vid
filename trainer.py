import torch
import math
import torchvision
import numpy as np
import torch.nn.functional as F
from torch import nn
from models import AFE, CKD, HPE_EDE, MFE, Generator, Discriminator
from losses import PerceptualLoss, GANLoss, FeatureMatchingLoss, EquivarianceLoss, KeypointPriorLoss, HeadPoseLoss, DeformationPriorLoss
from utils import transform_kp, make_coordinate_grid_2d, apply_imagenet_normalization


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        self.idx_tensor = torch.FloatTensor(list(range(num_bins))).unsqueeze(0).cuda()
        self.n_bins = num_bins
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        real_yaw = self.fc_yaw(x)
        real_pitch = self.fc_pitch(x)
        real_roll = self.fc_roll(x)
        real_yaw = torch.softmax(real_yaw, dim=1)
        real_pitch = torch.softmax(real_pitch, dim=1)
        real_roll = torch.softmax(real_roll, dim=1)
        real_yaw = (real_yaw * self.idx_tensor).sum(dim=1)
        real_pitch = (real_pitch * self.idx_tensor).sum(dim=1)
        real_roll = (real_roll * self.idx_tensor).sum(dim=1)
        real_yaw = (real_yaw - self.n_bins // 2) * 3 * np.pi / 180
        real_pitch = (real_pitch - self.n_bins // 2) * 3 * np.pi / 180
        real_roll = (real_roll - self.n_bins // 2) * 3 * np.pi / 180

        return real_yaw, real_pitch, real_roll


class Transform:
    """
    Random tps transformation for equivariance constraints.
    reference: FOMM
    """

    def __init__(self, bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        self.control_points = make_coordinate_grid_2d((points_tps, points_tps))
        self.control_points = self.control_points.unsqueeze(0)
        self.control_params = torch.normal(mean=0, std=sigma_tps * torch.ones([bs, 1, points_tps ** 2]))

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:]).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, align_corners=True, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        control_points = self.control_points.type(coordinates.type())
        control_params = self.control_params.type(coordinates.type())
        distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
        distances = torch.abs(distances).sum(-1)

        result = distances ** 2
        result = result * torch.log(distances + 1e-6)
        result = result * control_params
        result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
        transformed = transformed + result

        return transformed


class GeneratorFull(nn.Module):
    def __init__(
        self,
        afe: AFE,
        ckd: CKD,
        hpe_ede: HPE_EDE,
        mfe: MFE,
        generator: Generator,
        discriminator: Discriminator,
        pretrained_path="hopenet_robust_alpha1.pkl",
        n_bins=66,
    ):
        super().__init__()
        pretrained = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], n_bins).cuda()
        pretrained.load_state_dict(torch.load(pretrained_path, map_location=torch.device("cpu")))
        for parameter in pretrained.parameters():
            parameter.requires_grad = False
        self.pretrained = pretrained
        self.afe = afe
        self.ckd = ckd
        self.hpe_ede = hpe_ede
        self.mfe = mfe
        self.generator = generator
        self.discriminator = discriminator
        self.weights = {
            "P": 10,
            "G": 1,
            "F": 10,
            "E": 20,
            "L": 10,
            "H": 20,
            "D": 5,
        }
        self.losses = {
            "P": PerceptualLoss(),
            "G": GANLoss(),
            "F": FeatureMatchingLoss(),
            "E": EquivarianceLoss(),
            "L": KeypointPriorLoss(),
            "H": HeadPoseLoss(),
            "D": DeformationPriorLoss(),
        }

    def forward(self, s, d):
        fs = self.afe(s)
        kp_c = self.ckd(s)
        transform = Transform(d.shape[0])
        transformed_d = transform.transform_frame(d)
        cated = torch.cat([s, d, transformed_d], dim=0)
        yaw, pitch, roll, t, delta = self.hpe_ede(cated)
        [t_s, t_d, t_tran], [delta_s, delta_d, delta_tran] = (
            torch.chunk(t, 3, dim=0),
            torch.chunk(delta, 3, dim=0),
        )
        with torch.no_grad():
            self.pretrained.eval()
            real_yaw, real_pitch, real_roll = self.pretrained(F.interpolate(apply_imagenet_normalization(cated), size=(224, 224)))
        [yaw_s, yaw_d, yaw_tran], [pitch_s, pitch_d, pitch_tran], [roll_s, roll_d, roll_tran] = (
            torch.chunk(yaw, 3, dim=0),
            torch.chunk(pitch, 3, dim=0),
            torch.chunk(roll, 3, dim=0),
        )
        kp_s, Rs = transform_kp(kp_c, yaw_s, pitch_s, roll_s, t_s, delta_s)
        kp_d, Rd = transform_kp(kp_c, yaw_d, pitch_d, roll_d, t_d, delta_d)
        transformed_kp, _ = transform_kp(kp_c, yaw_tran, pitch_tran, roll_tran, t_tran, delta_tran)
        reverse_kp = transform.warp_coordinates(transformed_kp[:, :, :2])
        deformation, occlusion = self.mfe(fs, kp_s, kp_d, Rs, Rd)
        generated_d = self.generator(fs, deformation, occlusion)
        output_d, features_d = self.discriminator(d, kp_d)
        output_gd, features_gd = self.discriminator(generated_d, kp_d)
        loss = {
            "P": self.weights["P"] * self.losses["P"](generated_d, d),
            "G": self.weights["G"] * self.losses["G"](output_gd, True, False),
            "F": self.weights["F"] * self.losses["F"](features_gd, features_d),
            "E": self.weights["E"] * self.losses["E"](kp_d, reverse_kp),
            "L": self.weights["L"] * self.losses["L"](kp_d),
            "H": self.weights["H"] * self.losses["H"](yaw, pitch, roll, real_yaw, real_pitch, real_roll),
            "D": self.weights["D"] * self.losses["D"](delta_d),
        }
        return loss, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion


class DiscriminatorFull(nn.Module):
    def __init__(self, discriminator: Discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.weights = {
            "G": 1,
        }
        self.losses = {
            "G": GANLoss(),
        }

    def forward(self, d, generated_d, kp_d):
        output_d, _ = self.discriminator(d, kp_d)
        output_gd, _ = self.discriminator(generated_d.detach(), kp_d)
        loss = {
            "G1": self.weights["G"] * self.losses["G"](output_gd, False, True),
            "G2": self.weights["G"] * self.losses["G"](output_d, True, True),
        }
        return loss
