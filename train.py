import os
import argparse
import torch.utils.data as data
import torch.multiprocessing as mp
from logger import Logger
from dataset import FramesDataset, DatasetRepeater
from distributed import init_seeds, init_dist


def main(proc, args):
    world_size = len(args.gpu_ids)
    init_seeds(not args.benchmark)
    init_dist(proc, world_size)
    trainset = DatasetRepeater(FramesDataset(), num_repeats=100)
    trainsampler = data.distributed.DistributedSampler(trainset)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=trainsampler)
    logger = Logger(args.ckp_dir, args.vis_dir, trainloader, args.lr)
    if args.ckp > 0:
        logger.load_cpk(args.ckp)
    for i in range(args.num_epochs):
        logger.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU")
    parser.add_argument("--benchmark", type=str2bool, default=True, help="Turn on CUDNN benchmarking")
    parser.add_argument("--gpu_ids", default=[0], type=eval, help="IDs of GPUs to use")
    parser.add_argument("--lr", default=0.00005, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data loader threads")
    parser.add_argument("--ckp_dir", type=str, default="ckp", help="Checkpoint dir")
    parser.add_argument("--vis_dir", type=str, default="vis", help="Visualization dir")
    parser.add_argument("--ckp", type=int, default=0, help="Checkpoint epoch")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    mp.spawn(main, nprocs=len(args.gpu_ids), args=(args,))
