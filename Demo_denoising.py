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

import random
import scipy.misc
import numpy

# Dummy function just to try running this
def Parameters_Setting(nSig):
	par = {'nSig':nSig/255}
	model = []
	return (par, model)


# set parameters
nSig = 50
[par, model]  =  Parameters_Setting( nSig )

# read clean image
I = scipy.misc.imread('cameraman.png')/255
par['I'] = I

# generate noisy image
random.seed()
# I doubt that this translates well to python
nim = I + par['nSig']*random.randrange(len(I))
par['nim'] = nim

PSNR_init = csnr(nim*255, I*255, 0, 0)
SSIM_init = cal_ssim(nim*255, I*255, 0, 0)
print('The initial value of PSNR = '+str(PSNR_init)+', SSIM = '+str(SSIM_init))

# PGPD denoising
[im_out, par] = PGPD_Denoising(par, model)
# [im_out,par]  =  PGPD_Denoising_faster(par,model) # faster speed

# calculate the PSNR and SSIM
PSNR_final = csnr(im_out*255, I*255, 0, 0)
SSIM_final = cal_ssim(im_out*255, I*255, 0, 0)
print('Cameraman Results: PSNR = '+str(PSNR_final)+', SSIM = '+str(SSIM_final))

