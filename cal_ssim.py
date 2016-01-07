import numpy
import scipy.ndimage
import skimage



def cal_ssim( im1, im2, b_row, b_col ):
   ssim  = 0
   h, w  =  im1.shape
   ch = 1
   '''if len(im1.shape) == 2:
      [n,m] = im1.shape
      ch = 1
   else:
      [n,m,ch] = im1.shape'''
   if ch == 1:
       mssim,ssim_map = ssim_index( im1[b_row+1:h-b_row, b_col+1:w-b_col], im2[ b_row+1:h-b_row, b_col+1:w-b_col] )
   else:
       for i in range(len(ch)):
           ssim = ssim + ssim_index( im1[b_row+1:h-b_row, b_col+1:w-b_col, i], im2[ b_row+1:h-b_row, b_col+1:w-b_col, i] )
       ssim = ssim/3
   return mssim

def ssim_index(img1, img2, K = None, window = None, L = None):
    ''' if (img2 == None && K == None && window == None && L == None):
	mssim = -float('inf')
	ssim_map = -float('inf')
        return mssim,ssim_map
    if (numpy.shape(img1) != numpy.shape(img2)):
        mssim= -float('inf')
        ssim_map= -float('inf')
        return mssim,ssim_map'''
    M,N=numpy.shape(img1)
    '''if (img1 != None && img2 != None):
        if ((M < 11) or (N < 11)):
            mssim = -float('inf')
            ssim_map = -float('inf')
            return mssim,ssim_map
        window=fspecial((11, 11), 1.5)
        K[0]=0.01
        K[1]=0.03
        L=255
    if (img1 != None && img2 != None && K != None):
        if ((M < 11) or (N < 11)):
            mssim = -float('inf')
            ssim_map = -float('inf')
            return mssim,ssim_map
        window=fspecial((11, 11), 1.5)
        L=255
        if (len(K) == 2):
            if (K[0] < 0 or K[1] < 0):
                mssim = -float('inf')
                ssim_map = -float('inf')
                return mssim,ssim_map
        else:
            mssim = -float('inf')
            ssim_map = -float('inf')
            return mssim,ssim_map'''   
    #if (img1 != None && img2 != None && K != None && window != None):
    window=fspecial((11, 11), 1.5)
    K = numpy.zeros(2)
    K[0]=0.01
    K[1]=0.03
    L=255
    H,W = numpy.shape(window)
    if ((H * W) < 4 or (H > M) or (W > N)):
        mssim = -float('inf')
        ssim_map = -float('inf')
        return mssim,ssim_map
    L=255
    if (len(K) == 2):
        if (K[0] < 0 or K[1] < 0):
            mssim = -float('inf')
            ssim_map = -float('inf')
            return mssim,ssim_map
    else:
        mssim = -float('inf')
        ssim_map = -float('inf')
        return mssim,ssim_map
    
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    window = window / sum(map(sum, window))
    img1 = skimage.img_as_float(img1)
    img2 = skimage.img_as_float(img2)
    mu1 = scipy.ndimage.filters.convolve(img1, window, mode = 'constant')
    mu1 = mu1[4:-5, 4:-5]
    mu2 = scipy.ndimage.filters.convolve(img2, window, mode = 'constant')
    mu2 = mu2[4:-5, 4:-5]
    mu1_sq=mu1*mu1
    mu2_sq=mu2*mu2
    mu1_mu2=mu1*mu2
    
    sigma1_sq = scipy.ndimage.filters.convolve(img1*(img1), window, mode = 'constant')
    sigma1_sq = sigma1_sq[4:-5, 4:-5]
    sigma1_sq = sigma1_sq - mu1_sq

    sigma2_sq = scipy.ndimage.filters.convolve(img2*(img2), window, mode = 'constant')
    sigma2_sq = sigma2_sq[4:-5, 4:-5]
    sigma2_sq = sigma2_sq - mu2_sq

    sigma12 = scipy.ndimage.filters.convolve(img1*(img2), window, mode = 'constant')
    sigma12 = sigma12[4:-5, 4:-5]
    sigma12 = sigma12 - mu1_mu2
    
    if (C1 > 0 and C2 > 0):
        ssim_map=((2. * mu1_mu2 + C1)*(2. * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    else:
        print 'hello'
        numerator1=2 * mu1_mu2 + C1
        numerator2=2 * sigma12 + C2
        denominator1=mu1_sq + mu2_sq + C1
        denominator2=sigma1_sq + sigma2_sq + C2
        ssim_map=numpy.ones(numpy.shape(mu1))
        index=(denominator1*(denominator2) > 0)
        ssim_map[index]=(numerator1[index]*(numerator2[index])) / (denominator1[index]*(denominator2[index]))
        index=(denominator1 != 0) and (denominator2 == 0)
        ssim_map[index]=numerator1[index] / denominator1[index]
    mssim=numpy.mean(ssim_map)
    return mssim,ssim_map

def fspecial(shape=(3,3),sigma=0.5):
    """
    http://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = numpy.ogrid[-m:m+1,-n:n+1]
    h = numpy.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < numpy.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
