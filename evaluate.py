import argparse
from models import AFE, CKD, HPE_EDE, MFE, Generator
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import os
from skimage import io, img_as_float32
from utils import transform_kp, transform_kp_with_new_pose


@torch.no_grad()
def eval(args):
    g_models = {"afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
    output_frames = []
    if args.source == "r":
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        s = np.array(video_array[0], dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        yaw_s, pitch_s, roll_s, t_s, delta_s = g_models["hpe_ede"](s)
        kp_s, Rs = transform_kp(kp_c, yaw_s, pitch_s, roll_s, t_s, delta_s)
        for img in video_array[1:]:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_d, Rd = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            # generated_d = F.interpolate(generated_d, scale_factor=0.5)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    elif args.source == "f":
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        for img in video_array:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            fs = g_models["afe"](img)
            kp_c = g_models["ckd"](img)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            kp_d, Rd = transform_kp_with_new_pose(kp_c, yaw, pitch, roll, t, delta, 0 * yaw, 0 * pitch, 0 * roll)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            # generated_d = F.interpolate(generated_d, scale_factor=0.5)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    else:
        s = img_as_float32(io.imread(args.source))[:, :, :3]
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        s = F.interpolate(s, size=(256, 256))
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        yaw, pitch, roll, t, delta = g_models["hpe_ede"](s)
        kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, delta)
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        for img in video_array:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_d, Rd = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    imageio.mimsave(args.output, output_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--ckp_dir", type=str, default="ckp", help="Checkpoint dir")
    parser.add_argument("--output", type=str, default="output.gif", help="Output video")
    parser.add_argument("--ckp", type=int, default=0, help="Checkpoint epoch")
    parser.add_argument("--source", type=str, default="r", help="Source image, f for face frontalization, r for reconstruction")
    parser.add_argument("--driving", type=str, help="Driving dir")
    parser.add_argument("--num_frames", type=int, default=90, help="Number of frames")

    args = parser.parse_args()
    eval(args)
