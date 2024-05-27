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
from utils import dict2namespace, seed_everything, get_train_test_loader
from degradation import SRDeg
from skimage import data
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from loss import MS_SSIM_L1_LOSS, PerceptualLoss, EarthMoverDisteLoss
import argparse

parser = argparse.ArgumentParser()

# options
parser.add_argument('--lr', required=True,type=float, help='learning rate')
parser.add_argument('--resume_checkpoint', type=int,help='resume checkpoint, if no, then None')
parser.add_argument('--step_num',type=int,help='number of step size(sampling number)')
parser.add_argument('--data_dir', type=str, help='define the data path')
parser.add_argument('--noise', type=bool, default = False, help='add noise to the input or not')
parser.add_argument('--workdir', type=str, help='define the work directory')
args = parser.parse_args()
lr = args.lr
resume_ckpt = args.resume_checkpoint
step_num = args.step_num
add_noise = args.noise
# options
dataset = "optic"
config_path = f"./configs/{dataset}.yml"
device = "cuda:0"
ema_rate = 0.9999
deg = "dehaze"
SR_factor = 1
noise_std = 0.01

# logging
work_dir = args.workdir
workdir = Path(f"{work_dir}/")
for t in ["y", "x0", "xt", "x0_pred"]:
    (workdir / "vis" / t).mkdir(exist_ok=True, parents=True)

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
train_dataloader, test_dataloader = get_train_test_loader(config,data_path)

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
if model.use_fp16:
    model.convert_to_fp16()
if resume_ckpt:
    print(resume_ckpt)
    model.load_state_dict(torch.load(f'{work_dir}/model_{str(resume_ckpt).zfill(4)}.pt', map_location=device))
model = model.to(device)
model.train()

# initialize ema model
ema = ExponentialMovingAverage(model.parameters(), decay=ema_rate)

# define optimizer, loss
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
loss_fn = nn.L1Loss()
loss_ms_ssim = MS_SSIM_L1_LOSS()
loss_perceptual = PerceptualLoss()
loss_emd = EarthMoverDisteLoss()

def IIR(x_1, t=1., N=100, delta=None, clip=False,input=None):
    y = x_1
    x_t = x_1
    torch_t = torch.tensor(np.ones(shape=(len(x_t), )),
                           dtype=torch.float32,
                           device=device)
    if not delta:
        delta = 1 / N
    xt_list = []
    pred_x0_list = []
    with torch.no_grad():
        xt_list.append(x_t.cpu().detach().numpy())
        x_t_delta = x_t
        for _ in tqdm(range(int(t / delta))):
            x_t = x_t_delta
            t_in = (torch_t * 1000).type(torch.long)

            t_tensor = torch_t.view(len(x_t), 1, 1, 1)

            pred_x0 = model(x_t, t_in)
            x_t_delta = (delta / t_tensor) * pred_x0 + (1 - (delta / t_tensor)) * x_t[:,:3]
            
            torch_t -= delta
            
            xt_list.append(x_t_delta[:,:3].cpu().detach().numpy())
            pred_x0_list.append(pred_x0.cpu().detach().numpy())
    return xt_list, pred_x0_list


print(f"Beginning training.")
# Do the training here
if resume_ckpt == None:
    resume_ckpt = 0
for epoch in range(int(resume_ckpt), int(resume_ckpt)+50000):
    num_iter = len(train_dataloader.dataset) // batch_size
    epoch_loss = 0
    modified_epoch_loss = 0
    cnt = 0
    for x0, x1 in tqdm(train_dataloader, total=num_iter):
        optimizer.zero_grad()
        x0 = x0.to(device)
        y = x1.to(device)
        t = torch.tensor(np.random.uniform(0, 1, size=(batch_size)), 
                         dtype=torch.float32,
                         device=device)
        
        if add_noise == False:
            xt = deg_model.sample_xt(x0, y, t)
        else:
            xt = deg_model.sample_xt_noise(x0,y,t)
    
        t_in = (t * 1000).type(torch.long)
        x0_pred = model(xt, t_in)
        step_loss = loss_fn(x0_pred, x0) + 0.5*loss_ms_ssim(x0_pred,x0) + 0.2*loss_perceptual(x0_pred,x0) + 0.5*loss_emd(x0_pred,x0)
        step_loss.backward()
        optimizer.step()
        # Update ema model
        ema.update()

        epoch_loss += step_loss
        cnt += 1
        
    # For every epoch, try sampling -- batch size 10
    print(f"Epoch {epoch} Loss {step_loss:.5f}")

    
    if epoch%10 == 0:

        print(f"Performing sampling...")

        # debugging purpose

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

            plt.imsave(f'{workdir}/val/{i}_xt_{psnr:.2f}.png',result1)
            plt.imsave(f'{workdir}/val/{i}_target.png',target)
            plt.imsave(f'{workdir}/val/{i}_input.png',input)

            plt.close()

            p += psnr
            i+=1
                

        print("total PSNR :", p/len(test_dataloader))

        # save (orig, ema) model
        if epoch != int(resume_ckpt):
            torch.save(model.state_dict(), str(workdir / f"model_{str(epoch).zfill(4)}.pt"))
            torch.save(ema.state_dict(), str(workdir / f"ema_{ema_rate}_{str(epoch).zfill(4)}.pt"))
