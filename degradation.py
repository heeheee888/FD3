import torch

from resizer import Resizer


class PairedDeg:
    def __init__(self, noise_std, img_size, channels, device):
        self.noise_std = noise_std
        self.img_size = img_size
        self.channels = channels
        self.device = device
        
    def sample_noise_t(self, x0, t):
        eps = torch.randn_like(x0) * self.noise_std
        eps_t = eps
        return eps_t
    
    def A(self, x0):
        pass
        
    def sample_xt(self, x0, y, t):
        t_tensor = t.view(len(x0), 1, 1, 1)
        interp = (1 - t_tensor) * x0 + t_tensor * y
        noise = self.sample_noise_t(x0, t)
        return interp

    def sample_xt_noise(self, x0, y, t):
        t_tensor = t.view(len(x0), 1, 1, 1)
        interp = (1 - t_tensor) * x0 + t_tensor * y
        noise = self.sample_noise_t(x0, t)
        return interp + noise * t_tensor
    
    
    
class SRDeg(PairedDeg):
    def __init__(self, noise_std,img_size, channels, device, 
                 SR_factor=4, batch_size=16):
        super().__init__(noise_std, img_size, channels, device)
        self.SR_factor = SR_factor
        self.batch_size = batch_size
        shape = (batch_size, channels, img_size, img_size)
        shape_d = (batch_size, channels, img_size // SR_factor, img_size // SR_factor)
        self.down = Resizer(shape, 1 / SR_factor).to(device)
        self.up = Resizer(shape_d, SR_factor).to(device)
        
    # get y from x0
    def A(self, x0):
        return self.up(self.down(x0))