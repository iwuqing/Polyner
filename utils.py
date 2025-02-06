# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def psnr(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return peak_signal_noise_ratio(ground_truth, image, data_range=data_range)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)


def fan_beam_ray(proj_pos, SOD):
    origin_x = 0
    origin_y = -1
    y = np.linspace(-1, 1, int(2*SOD)).reshape(-1, 1)  # (2*SOD, ) -> (2*SOD, 1)
    x = np.zeros_like(y)  # (2*SOD, 1)
    xy_temp = np.concatenate((x, y), axis=-1)  # (2*SOD, 2)
    xy_temp = np.concatenate((xy_temp, np.ones_like(x)), axis=-1)  # (2*SOD, 3)
    num_det = len(proj_pos)
    xy = np.zeros(shape=(num_det, int(2*SOD), 2)) # (L, 2*SOD, 2)
    for i in range(num_det):
        fan_angle_rad = np.deg2rad(proj_pos[num_det-i-1])
        M = np.array(
            [
                [np.cos(fan_angle_rad), -np.sin(fan_angle_rad),
                 -1*origin_x*np.cos(fan_angle_rad)+origin_y*np.sin(fan_angle_rad)+origin_x],
                [np.sin(fan_angle_rad), np.cos(fan_angle_rad),
                 -1*origin_x*np.sin(fan_angle_rad)-origin_y*np.cos(fan_angle_rad)+origin_y],
                [0, 0, 1]
            ]
        )
        temp = xy_temp @ M.T # (2*SOD, 3) @ (3, 3) -> (2*SOD, 3)
        xy[i, :, :] = temp[:, :2] # (2*SOD, 2)
    return xy


def grid_coordinate(h, w):
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    x, y = np.meshgrid(x, y, indexing='ij')  # (h, w), (h, w)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (h*w, 2)
    return xy


def rotate_ray(xy, angle):
    xy_shape = xy.shape
    angle_rad = np.deg2rad(angle)
    trans_mat = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)],
        ]
    )
    xy = xy.reshape(-1, 2)
    xy = (np.dot(xy, trans_mat.T)).reshape(xy_shape)
    return xy