# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attenuation_Smootion_Over_Energies_Loss(nn.Module):
    def __init__(self, mask, lamb):
        super(Attenuation_Smootion_Over_Energies_Loss, self).__init__()
        self.mask = mask
        self.lamb = lamb
    def forward(self, ray, intensity):
        batch_size, num_sample_ray, k, e_level = intensity.shape
        mask = F.grid_sample(
            self.mask, ray.unsqueeze(0).unsqueeze(0), mode='nearest', align_corners=False
        )[0, 0, 0, :].view(batch_size, num_sample_ray, k)   # (batch_size, num_sample_ray, 2*SOD)
        diff = torch.sum(torch.abs(intensity[:, :, :, 1:] - intensity[:, :, :, :e_level-1]), dim=-1) * mask
        return self.lamb * torch.sum(diff) / (batch_size * num_sample_ray * k)
