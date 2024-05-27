import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import os

from pathlib import Path
from tqdm import tqdm
from torch import nn
from torchvision.utils import save_image, make_grid
from torch_ema import ExponentialMovingAverage
from PIL import Image

from guided_diffusion.script_util import create_model, args_to_dict
from guided_diffusion import dist_util, logger
from utils import dict2namespace, seed_everything, get_test_loader
from degradation import SRDeg
from guided_diffusion.image_datasets import load_data, ImageDataset

from skimage import data
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from loss import MS_SSIM_L1_LOSS, PerceptualLoss, EarthMoverDisteLoss

from torch.autograd import grad

import argparse

parser = argparse.ArgumentParser()

# options
parser.add_argument('--lr', required=True,type=float, help='learning rate')
parser.add_argument('--resume_checkpoint', type=int,help='resume checkpoint, if no, then None')
parser.add_argument('--step_num',type=int,help='number of step size(sampling number)')
parser.add_argument('--data_dir', type=str, help='define the data path')
args = parser.parse_args()
lr = args.lr
resume_ckpt = args.resume_checkpoint
step_num = args.step_num
device = "cuda:0"
ema_rate = 0.9999
dataset = "optic"
config_path = f"./configs/{dataset}.yml"
deg = "dehaze"
SR_factor = 4
noise_std = 0.01

# logging
workdir = Path(f"./result/{dataset}_{step_num}_{str(resume_ckpt).zfill(4)}_optic")
workdir.mkdir(exist_ok=True, parents=True)

# parse config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)

# for convenience
img_size = config.model.image_size
channels = config.data.channels
batch_size = config.data.batch_size

# fix seed
seed_everything(42)


# define data loading pipeline
data_path = args.data_dir
test_dataloader = get_test_loader(config,data_path)

# degradation model
deg_model = SRDeg(noise_std, 
                  img_size, 
                  channels,
                  device,
                  SR_factor=SR_factor)


# define model
print(f"Initializing models")
config_dict = vars(config.model)
model = create_model(**config_dict)

dist_util.setup_dist()
if resume_ckpt:
    model.load_state_dict(dist_util.load_state_dict(f'./workdir/model_{str(resume_ckpt).zfill(4)}.pt', map_location='cpu'))
model = model.to(device)
model.eval()

# initialize ema model
ema = ExponentialMovingAverage(model.parameters(), decay=ema_rate)
ema.load_state_dict(dist_util.load_state_dict(f'./workdir/ema_0.9999_{str(resume_ckpt).zfill(4)}.pt', map_location='cpu'))


# define optimizer, loss
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.L1Loss()
loss_ms_ssim = MS_SSIM_L1_LOSS()
loss_perceptual = PerceptualLoss()
loss_emd = EarthMoverDisteLoss()

def IIR(x_1, t=1., N=100, delta=None, clip=False,gt=None, input=None):
    y = x_1
    x_t = x_1
    torch_t = torch.tensor(np.ones(shape=(len(x_t), )),
                           dtype=torch.float32,
                           device=device)
    if not delta:
        delta = 1 / N
    xt_list = []
    pred_x0_list = []

    step = []


    with torch.no_grad():
        with ema.average_parameters():
            xt_list.append(x_t.cpu().detach().numpy())
            x_t_delta = x_t
            
            for num in tqdm(range(int(t / delta))):

                x_t = x_t_delta
                t_in = (torch_t * 1000).type(torch.long)

                t_tensor = torch_t.view(len(x_t), 1, 1, 1)

                pred_x0 = model(x_t, t_in)
                if clip:
                    pred_x0_clip = torch.clip(pred_x0, 0.0, 1.0)

                x_t_delta = (delta / t_tensor) * pred_x0_clip + (1 - (delta / t_tensor)) * x_t[:,:3]
                x_t_delta_clip = torch.clip(x_t_delta,0.0,1.0)
                step.append(delta/torch_t.to('cpu'))
                torch_t -= delta

                xt_list.append(x_t_delta_clip[:,:3].cpu().detach().numpy())
                pred_x0_list.append(pred_x0_clip.cpu().detach().numpy())

        return xt_list, pred_x0_list


print(f"Performing sampling...")
p = 0
i = 0
for test_loader in test_dataloader:

    # debugging purpose
    x0, x1 = test_loader
    x0 = x0.to(device)
    y = x1.to(device)

    torch_t = torch.tensor(np.ones(shape=(len(x0), )),
                            dtype=torch.float32,
                            device=device)
        
    x_1 = deg_model.sample_xt(x0, y, torch_t)
    xt_list, pred_x0_list = IIR(x_1, N=step_num,clip=True,gt=x0,input=y)

    target = np.array(x0.to('cpu'))
    target = np.transpose(target[0],(1,2,0))
    input = np.array(x1.to('cpu'))
    input = np.transpose(input[0],(1,2,0))
    result1 = np.array(xt_list[step_num][0])
    result1 = np.transpose(result1,(1,2,0))
    result1= np.clip(result1, 0.0,1.0)


    # calculate PSNR
    result = np.array(xt_list[step_num][0])
    result = np.transpose(result,(1,2,0))
    psnr = peak_signal_noise_ratio(target,result)
    print(psnr)

    plt.imsave(f'{workdir}/{i}_xt_{psnr:.2f}.png',result1)
    plt.imsave(f'{workdir}/{i}_target.png',target)
    plt.imsave(f'{workdir}/{i}_input.png',input)

    plt.close()

    p += psnr
    i+=1

print("PSNR :", p/len(test_dataloader))
