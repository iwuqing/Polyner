# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import numpy as np
import utils
import SimpleITK as sitk

if __name__ == '__main__':

    p = []
    s = []
    for i in range(10):
        metal = sitk.GetArrayFromImage(sitk.ReadImage('input/mask_{}.nii'.format(i)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage('input/gt_{}.nii'.format(i)))
        recon = sitk.GetArrayFromImage(sitk.ReadImage('output/polyner_{}.nii'.format(i)))

        gt = np.where(metal==1, 0, gt)
        recon = np.where(metal==1, 0, recon)

        p.append(utils.psnr(image=recon, ground_truth=gt))
        s.append(utils.ssim(image=recon, ground_truth=gt))

    print('Polyner [PSNR]:{}±{}'.format(np.round(np.mean(p), 2), np.round(np.std(p), 2)))
    print('Polyner [SSIM]:{}±{}'.format(np.round(np.mean(s), 4), np.round(np.std(s), 4)))


    p = []
    s = []
    for i in range(10):
        metal = sitk.GetArrayFromImage(sitk.ReadImage('input/mask_{}.nii'.format(i)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage('input/gt_{}.nii'.format(i)))
        recon = sitk.GetArrayFromImage(sitk.ReadImage('input/ma_{}.nii'.format(i)))

        gt = np.where(metal==1, 0, gt)
        recon = np.where(metal==1, 0, recon)

        p.append(utils.psnr(image=recon, ground_truth=gt))
        s.append(utils.ssim(image=recon, ground_truth=gt))

    print('FBP [PSNR]:{}±{}'.format(np.round(np.mean(p), 2), np.round(np.std(p), 2)))
    print('FBP [SSIM]:{}±{}'.format(np.round(np.mean(s), 4), np.round(np.std(s), 4)))