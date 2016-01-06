#------------------------------------------------------------------------------------------------
# PGPD_Denoising - Denoising by Weighted Sparse Coding 
#                                with Learned Patch Group Prior.
# CalNonLocal - Calculate the non-local similar patches (Noisy Patch Groups)
# Author:  Jun Xu, csjunxu@comp.polyu.edu.hk
#              The Hong Kong Polytechnic University
#------------------------------------------------------------------------------------------------

# Translated to Python by Duncan Sommer and Kaisar Kushibar

import numpy
import csnr
import cal_ssim


def PGPD_Denoising(par, model):
    im_out = par['nim']
    h, w =im_out.shape
    
    # Fill in more parameters
    par['nSig0'] = par['nSig']
    par['maxr'] = h-par['ps']+1
    par['maxc'] = w-par['ps']+1
    par['maxrc'] = par['maxr'] * par['maxc']
    par['h'] = h
    par['w'] = w
    
    r = range(0, par['maxr'], par['step'])
    par['r'] = r + range(r[-1]+1, par['maxr'])
    # Unsure if I translated this line correctly
    #par['r'] = [r r()+1:par['maxr']]
    c = range(0, par['maxc'], par['step'])
    par['c'] = c + range(c[-1]+1, par['maxc'])

    par['lenr'] = len(par['r'])
    par['lenc'] = len(par['c'])
    par['lenrc'] = par['lenr']*par['lenc']
    par['ps2'] = par['ps']*par['ps']


    for ite in range(par['IteNum']):
        # iterative regularization
        im_out = im_out + par['delta']*(par['nim'] - im_out)

        # estimation of noise variance
        if ite == 0:
            par['nSig'] = par['nSig0']
        else:
            dif = numpy.mean(numpy.mean( numpy.square(par['nim']-im_out) )) 
            par['nSig'] = numpy.sqrt( abs(par['nSig0']*par['nSig0']-dif) )*par['eta']


        # search non-local patch groups
        [nDCnlX,blk_arr,DC,par] = CalNonLocal(im_out, par)
        # Gaussian dictionary selection by MAP
        if (ite - 1)%2 == 0:
               # CHECKPOINT
            PYZ = numpy.zeros(model['nmodels'],(DC.shape)[1])
            sigma2I = (par['nSig']**2)*numpy.eye(par['ps2'])
            for i in range(model['nmodels']):
                sigma = model['covs'][:,:,i] + sigma2I
                R,tmp = numpy.linalg.cholesky(sigma)
                Q = numpy.linalg.lstsq(numpy.transpose(R),nDCnlX)
                TempPYZ = -numpy.sum(numpy.log(numpy.diag(R))) - numpy.dot(Q,Q)/2.
                TempPYZ = numpy.reshape(TempPYZ,[par['nlsp'] (DC.shape)[1]])
                PYZ[i,:] = numpy.sum(TempPYZ)
            
            # find the most likely component for each patch group
            tmp, dicidx = max(PYZ)
            dicidx = numpy.transpose(dicidx)
            idx, s_idx = numpy.sort(dicidx)
            idx2 = idx[:-1] - idx[1:]
            seq = numpy.nonzero(idx2)
            seg = numpy.concatenate([0, seq, len(dicidx)])
            seg = seg.reshape(len(seg), 1)
             
        # Weighted Sparse Coding
        X_hat = numpy.zeros(par['ps2'],par['maxrc'])
        W = numpy.zeros(par['ps2'],par['maxrc'])
        for j in range(len(seg)-2):
            idx =   s_idx[seg[j]+range(seg[j+1])]
            cls =   dicidx[idx[0]]
            D   =   par['D'][:,:, cls]
            S    = par['S'][:,cls]
            lambdaM = numpy.tile((par['c1']*par['nSig']**2.)/(numpy.sqrt(S)+0.0001 ), par['nlsp'])
            for i in range(len(idx)):
                #Y = nDCnlX(:,(idx(i)-1)*par['nlsp']+1:idx(i)*par['nlsp'])
                Y = nDCnlX[:,(idx[i]-1)*par['nlsp']+range(idx[i]*par['nlsp'])]
                b = numpy.transpose(D)*Y
                # soft threshold
                alpha = numpy.sign(b)*numpy.max(abs(b)-lambdaM/2.,0.)
                # add DC components and aggregation
                #X_hat[:,blk_arr[:,idx[i]]] = X_hat[:,blk_arr[:,idx[i]]]+bsxfun(@plus,D*alpha, DC[:,idx[i]])
                X_hat[:,blk_arr[:,idx[i]]] = X_hat[:,blk_arr[:,idx[i]]]+(D*alpha + DC[:,idx[i]])
                W[:,blk_arr[:,idx[i]]]=W[:,blk_arr[:,idx[i]]] + numpy.ones(par['ps2'], par['nlsp'])
        # Reconstruction
        im_out = numpy.zeros(h,w)
        im_wei = numpy.zeros(h,w)
        r = range(par['maxr'])
        c = range(par['maxc'])
        k = 0
        for i in range(par['ps']):
            for j in range(par['ps']):
                #im_out(r-1+i,c-1+j)  =  im_out(r-1+i,c-1+j) + reshape( numpy.transpose(X_hat(k,:)), [par['maxr'] par['maxc']])
                #im_wei(r-1+i,c-1+j)  =  im_wei(r-1+i,c-1+j) + reshape( numpy.transpose(W(k,:)), [par['maxr'] par['maxc']])
                 im_out[r + i,c + j]=im_out[r + i,c + j] + numpy.reshape(numpy.transpose(X_hat[k,:]),[par['maxr'],par['maxc']])
                 im_wei[r + i,c + j]=im_wei[r + i,c + j] + numpy.reshape(numpy.transpose(W[k,:]),[par['maxr'],par['maxc']])
                 k = k+1
            
        
        im_out  =  im_out / im_wei
        # calculate the PSNR and SSIM
        PSNR = csnr.csnr( im_out*255, par['I']*255., 0, 0 )
        SSIM = cal_ssim.cal_ssim( im_out*255, par['I']*255., 0, 0 )
        print('Iter #d : PSNR = #2.4f, SSIM = #2.4f\n',ite, PSNR,SSIM)
    
    im_out[im_out > 1] = 1
    im_out[im_out < 0] = 0
    return im_out, par
def CalNonLocal(im,par):
    #im=single(im)
    X=numpy.zeros(par['ps2'],par['maxrc'])
    k=0
    for i in range(par['ps']):
        for j in range(par['ps']):
            blk=im[i : -1-par['ps']+i, j : -1-par['ps']+j]
            X[k,:]=numpy.transpose(blk[:])
            k = k + 1
            
    Index=(range(par['maxrc']))
    Index=numpy.reshape(Index, par['maxr'], par['maxc'])
    blk_arr=numpy.zeros(par['nlsp'], par['lenrc'])
    DC=numpy.zeros(par['ps2'],par['lenrc'])
    nDCnlX=numpy.zeros(par['ps2'], par['lenrc'] * par['nlsp'])
    for i in range(par['lenr']):
        for j in range(par['lenc']):
            row=(par['r'])[i]
            col=(par['c'])[j]
            off=(col - 1) * par['maxr'] + row
            off1=(j - 1) * par['lenr'] + i
            rmin=numpy.max(row - par['Win'], 1)
            rmax=numpy.min(row + par['Win'], par['maxr'])
            cmin=numpy.max(col - par['Win'], 1)
            cmax=numpy.min(col + par['Win'], par['maxc'])
            idx=Index[rmin:rmax,cmin:cmax]
            idx=idx[:]
            neighbor=X[:,idx]
            seed=X[:,off]
            dis=numpy.sum((neighbor-seed)**2)
            tmp,ind = numpy.sort(dis)
            indc=idx[ind[0:par['nlsp']]]
            blk_arr[:,off1]=indc
            temp=X[:,indc]
            DC[:,off1]=numpy.mean(temp, 2)
            nDCnlX[:,(off1 - 1) * par['nlsp'] + range(off1) * par['nlsp']] = temp - DC[:,off1]
    return nDCnlX,blk_arr,DC,par
