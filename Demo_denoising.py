#--------------------------------------------------------------------------------------------------
# This is an implementation of the PGPD algorithm for image denoising.
# Author:  Jun Xu, csjunxu@comp.polyu.edu.hk
#              The Hong Kong Polytechnic University
# Please refer to the following paper if you use this code:
# Jun Xu, Lei Zhang, Wangmeng Zuo, David Zhang, and Xiangchu Feng,
# Patch Group Based Nonlocal Self-Similarity Prior Learning for Image Denoising.
# IEEE Int. Conf. Computer Vision (ICCV), Santiago, Chile, December 2015.
# Please see the file License.txt for the license governing this code.
#--------------------------------------------------------------------------------------------------

# Translated to Python by Duncan Sommer

import scipy.misc
import skimage
import skimage.io
import cal_ssim
import csnr
import PGPD_Denoising
import numpy
import Parameters_Setting

# set parameters
nSig = 50.
[par, model]  =  Parameters_Setting.Parameters_Setting( nSig )

# read clean image
I = skimage.io.imread('cameraman.png')/255.
par['I'] = I
# generate noisy image
#random.seed()
# I doubt that this translates well to python
#par.nim =   par.I + par.nSig*randn(size(par.I));
nim = I + par['nSig']*numpy.random.randn(I.shape[0], I.shape[1])
par['nim'] = nim

PSNR_init = csnr.csnr(nim*255., I*255., 0., 0.)
SSIM_init = cal_ssim.cal_ssim(nim*255., I*255., 0., 0.)
print('The initial value of PSNR = '+str(PSNR_init)+', SSIM = '+str(SSIM_init))

# PGPD denoising
[im_out, par] = PGPD_Denoising.PGPD_Denoising(par, model)
# [im_out,par]  =  PGPD_Denoising_faster(par,model) # faster speed

# calculate the PSNR and SSIM
PSNR_final = csnr.csnr(im_out*255., I*255., 0., 0.)
SSIM_final = cal_ssim.cal_ssim(im_out*255., I*255., 0., 0.)
print('Cameraman Results: PSNR = '+str(PSNR_final)+', SSIM = '+str(SSIM_final))