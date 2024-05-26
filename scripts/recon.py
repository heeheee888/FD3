import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from pathlib import Path
from utils import clear_color, normalize_np, mask_generator, clear, prepare_im
import math
from resizer import Resizer
from functools import partial
from tqdm import tqdm



# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    out_path = Path(args.save_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_kwargs = {}
    
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        grayscale=args.grayscale,
        paired=args.paired,
    )
    
    # degradation operator
    down_N = 2
    shape = (args.batch_size, 1, args.image_size, args.image_size)
    shape_d = (args.batch_size, 1, int(args.image_size / down_N), int(args.image_size / down_N))
    down = Resizer(shape, 1 / down_N).to(dist_util.dev())
    up = Resizer(shape_d, down_N).to(dist_util.dev())

    for j in range(args.num_samples):
        out_path_j = out_path / f"{str(j).zfill(3)}"
        for ty in ["x0hat", "xt"]:
            (out_path_j / ty).mkdir(exist_ok=True, parents=True)
        # load data
        x0, _ = next(data)
        x0 = x0.to(dist_util.dev())
        cond = {}
        # make degradation on the fly
        y = up(down(x0))
        
        plt.imsave(str(out_path_j / f"y.png"), clear(y))
        plt.imsave(str(out_path_j / f"x0.png"), clear(x0))
        
        # sample xt
        for t in tqdm(reversed(range(0, 1000, 20))):
            t_in = th.ones([1], device=dist_util.dev()) * t
            t_in = t_in.type(th.long)
            xt = diffusion.q_sample(x0, y, t_in)
            x0hat = model(xt, t_in)
            
            plt.imsave(str(out_path_j / "x0hat" / f"{str(t).zfill(4)}.png"), clear(x0hat))
            plt.imsave(str(out_path_j / "xt" / f"{str(t).zfill(4)}.png"), clear(xt))
            

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=1,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        grayscale=True,
        image_size=32,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory where the results will be saved')
    parser.add_argument('--data_dir', type=str, default='MNIST')
    parser.add_argument('--paired', action="store_true")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()