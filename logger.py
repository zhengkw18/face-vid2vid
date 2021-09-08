import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
from distributed import master_only, master_only_print, get_rank, is_master
from models import AFE, CKD, HPE_EDE, MFE, Generator, Discriminator
from trainer import GeneratorFull, DiscriminatorFull
from tqdm import tqdm


def to_cpu(losses):
    return {key: value.detach().data.cpu().numpy() for key, value in losses.items()}


class Logger:
    def __init__(
        self,
        ckp_dir,
        vis_dir,
        dataloader,
        lr,
        checkpoint_freq=1,
        visualizer_params={"kp_size": 5, "draw_border": True, "colormap": "gist_rainbow"},
        zfill_num=8,
        log_file_name="log.txt",
    ):

        self.g_losses, self.d_losses = [], []
        self.ckp_dir = ckp_dir
        self.vis_dir = vis_dir
        if is_master():
            if not os.path.exists(self.ckp_dir):
                os.makedirs(self.ckp_dir)
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
            self.log_file = open(log_file_name, "a")
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float("inf")
        self.g_models = {"afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
        self.d_models = {"discriminator": Discriminator()}
        for name, model in self.g_models.items():
            self.g_models[name] = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[get_rank()])
        for name, model in self.d_models.items():
            self.d_models[name] = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[get_rank()])
        self.g_optimizers = {name: torch.optim.Adam(self.g_models[name].parameters(), lr=lr, betas=(0.5, 0.999)) for name in self.g_models.keys()}
        self.d_optimizers = {name: torch.optim.Adam(self.d_models[name].parameters(), lr=lr, betas=(0.5, 0.999)) for name in self.d_models.keys()}
        self.g_full = GeneratorFull(**self.g_models, **self.d_models)
        self.d_full = DiscriminatorFull(**self.d_models)
        self.g_loss_names, self.d_loss_names = None, None
        self.dataloader = dataloader

    def __del__(self):
        self.save_cpk()
        if is_master():
            self.log_file.close()

    @master_only
    def log_scores(self):
        loss_mean = np.array(self.g_losses).mean(axis=0)
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.g_loss_names, loss_mean)])
        loss_string = "G" + str(self.epoch).zfill(self.zfill_num) + ") " + loss_string
        print(loss_string, file=self.log_file)
        self.g_losses = []
        loss_mean = np.array(self.d_losses).mean(axis=0)
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.d_loss_names, loss_mean)])
        loss_string = "D" + str(self.epoch).zfill(self.zfill_num) + ") " + loss_string
        print(loss_string, file=self.log_file)
        self.d_losses = []
        self.log_file.flush()

    @master_only
    def visualize_rec(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
        image = self.visualizer.visualize(s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion)
        imageio.imsave(os.path.join(self.vis_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    @master_only
    def save_cpk(self):
        ckp = {
            **{k: v.module.state_dict() for k, v in self.g_models.items()},
            **{k: v.module.state_dict() for k, v in self.d_models.items()},
            **{"optimizer_" + k: v.state_dict() for k, v in self.g_optimizers.items()},
            **{"optimizer_" + k: v.state_dict() for k, v in self.d_optimizers.items()},
            "epoch": self.epoch,
        }
        ckp_path = os.path.join(self.ckp_dir, "%s-checkpoint.pth.tar" % str(self.epoch).zfill(self.zfill_num))
        torch.save(ckp, ckp_path)

    def load_cpk(self, epoch):
        ckp_path = os.path.join(self.ckp_dir, "%s-checkpoint.pth.tar" % str(epoch).zfill(self.zfill_num))
        checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
        for k, v in self.g_models.items():
            v.module.load_state_dict(checkpoint[k])
        for k, v in self.d_models.items():
            v.module.load_state_dict(checkpoint[k])
        for k, v in self.g_optimizers.items():
            v.load_state_dict(checkpoint["optimizer_" + k])
        for k, v in self.d_optimizers.items():
            v.load_state_dict(checkpoint["optimizer_" + k])
        self.epoch = checkpoint["epoch"] + 1

    @master_only
    def log_iter(self, g_losses, d_losses):
        g_losses = collections.OrderedDict(g_losses.items())
        d_losses = collections.OrderedDict(d_losses.items())
        if self.g_loss_names is None:
            self.g_loss_names = list(g_losses.keys())
        if self.d_loss_names is None:
            self.d_loss_names = list(d_losses.keys())
        self.g_losses.append(list(g_losses.values()))
        self.d_losses.append(list(d_losses.values()))

    @master_only
    def log_epoch(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores()
        self.visualize_rec(s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion)

    def step(self):
        master_only_print("Epoch", self.epoch)
        with tqdm(total=len(self.dataloader.dataset)) as progress_bar:
            for s, d in self.dataloader:
                s = s.cuda(non_blocking=True)
                d = d.cuda(non_blocking=True)
                for optimizer in self.g_optimizers.values():
                    optimizer.zero_grad()
                losses_g, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion = self.g_full(s, d)
                loss_g = sum(losses_g.values())
                loss_g.backward()
                for optimizer in self.g_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                for optimizer in self.d_optimizers.values():
                    optimizer.zero_grad()
                losses_d = self.d_full(d, generated_d, kp_d)
                loss_d = sum(losses_d.values())
                loss_d.backward()
                for optimizer in self.d_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                self.log_iter(to_cpu(losses_g), to_cpu(losses_d))
                if is_master():
                    progress_bar.update(len(s))
        self.log_epoch(s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion)
        self.epoch += 1


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap="gist_rainbow"):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
        images = []
        # Source image with keypoints
        source = s.data.cpu()
        kp_source = kp_s.data.cpu().numpy()[:, :, :2]
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        transformed = transformed_d.data.cpu().numpy()
        transformed = np.transpose(transformed, [0, 2, 3, 1])
        transformed_kp = transformed_kp.data.cpu().numpy()[:, :, :2]
        images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = kp_d.data.cpu().numpy()[:, :, :2]
        driving = d.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Result with and without keypoints
        prediction = generated_d.data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        # Occlusion map
        occlusion_map = occlusion.data.cpu().repeat(1, 3, 1, 1)
        occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
        occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
        images.append(occlusion_map)

        image = self.create_image_grid(*images)
        image = image.clip(0, 1)
        image = (255 * image).astype(np.uint8)
        return image
