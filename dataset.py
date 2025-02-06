# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data

class TrainData(data.Dataset):
    def __init__(self, proj_path, proj_pos_path, num_sample_ray, num_angle, SOD, voxel_size):
        self.num_angle = num_angle
        self.num_sample_ray = num_sample_ray
        self.SOD = SOD
        self.voxel_size = voxel_size
        self.angles = np.linspace(0., 360., num=self.num_angle, endpoint=False)  # (num_angle, )
        self.proj_pos = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path)).reshape(-1) # (num_det, )
        self.num_det = len(self.proj_pos)
        # projection, i.e., sinogram & metal_trace
        self.proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))  # (num_angle, num_det)
        # ray
        self.rays = utils.fan_beam_ray(self.proj_pos, self.SOD) # (num_det, 2*SOD, 2)
        self.index_max = self.num_det - self.num_sample_ray

    def __getitem__(self, item):
        ang = self.angles[item]
        proj = self.proj[item].reshape(-1, )  # (num_det, )
        # sample ray, projection, and metal trace
        index = np.random.randint(0, self.index_max, size=1)[0]
        ray_sample = self.rays[index:index+self.num_sample_ray]     # (num_sample_ray, 2*SOD, 2)
        proj_sample = proj[index:index+self.num_sample_ray]     # (num_sample_ray, )
        # rotate ray
        ray_sample = utils.rotate_ray(xy=ray_sample, angle=ang)
        return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData(data.Dataset):
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.xy = utils.grid_coordinate(h=self.h, w=self.w).reshape(1, int(h*w), 2)

    def __getitem__(self, item):
        return self.xy[item]    # (h*w, 2)

    def __len__(self):
        return 1