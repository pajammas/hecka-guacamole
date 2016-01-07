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
        if (ite)%2 == 0:
               # CHECKPOINT
            PYZ = numpy.zeros((model['nmodels'],(DC.shape)[1]))
            sigma2I = (par['nSig']**2)*numpy.eye(par['ps2'])
            for i in range(model['nmodels']):
                sigma = model['covs'].item()[:,:,i] + sigma2I
                R = numpy.linalg.cholesky(sigma)
                Q = numpy.transpose(numpy.linalg.lstsq(numpy.transpose(R),nDCnlX)[0])
                #print numpy.diag(R).shape, numpy.log(numpy.diag(R)).shape
                print numpy.sum(numpy.log(numpy.diag(R))).shape
                print (numpy.dot(numpy.transpose(Q),Q)/2.).shape
                #print Q[0].shape, Q[1].shape
                #print type(numpy.dot(Q,Q)/2.), numpy.sum(numpy.log(numpy.diag(R)))
                TempPYZ = -numpy.sum(numpy.log(numpy.diag(R))) - numpy.dot(numpy.transpose(Q),Q)/2.
                print DC.shape, TempPYZ.shape, par['nlsp']
                TempPYZ = TempPYZ.reshape((par['nlsp'], DC.shape[1]))
                #print TempPYZ.shape
                #print numpy.sum(TempPYZ)
                
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
        X_hat = numpy.zeros((par['ps2'],par['maxrc']))
        W = numpy.zeros((par['ps2'],par['maxrc']))
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
    X=numpy.zeros((int(par['ps2']), int(par['maxrc'])))
    k = 0
    for i in range(par['ps']):
        for j in range(par['ps']):
            blk=im[i-par['ps']+i, j-par['ps']+j]
            X[k,:]=numpy.transpose(blk)
            k = k + 1
            
    Index=(range(par['maxrc']))
    
    Index=numpy.reshape(Index, (par['maxr'], par['maxc']))
    blk_arr=numpy.zeros((par['nlsp'], par['lenrc']))
    DC=numpy.zeros((par['ps2'],par['lenrc']))
    nDCnlX=numpy.zeros((par['ps2'], par['lenrc'] * par['nlsp']))
    for i in range(par['lenr']):
        for j in range(par['lenc']):
            row=(par['r'])[i]
            col=(par['c'])[j]
            # col-1, j-1, +i;
            off=(col) * par['maxr'] + row + 1
            off1=(j) * par['lenr'] + i + 1
            rmin=numpy.maximum(row - par['Win'], 0)
            rmax=numpy.minimum(row + par['Win'], par['maxr'])
            cmin=numpy.maximum(col - par['Win'], 0)
            cmax=numpy.minimum(col + par['Win'], par['maxc'])
            idx=Index[rmin:rmax+1,cmin:cmax+1]
            neighbor=X[:,idx.reshape(1,(idx.shape)[0]*(idx.shape)[1])[0]]
            #neighbor = neighbor.reshape((neighbor.shape)[0], (neighbor.shape)[1]*(neighbor.shape)[2])
            seed=X[:,off-1]
            dis=numpy.sum((numpy.transpose(neighbor)-seed)**2, 1)
            #dis=numpy.sum((neighbor-seed)**2, 1)
            ind = numpy.argsort(dis)
            idx = idx.reshape((idx.shape)[0]*(idx.shape)[1], 1)
            indc=idx[ind[0:par['nlsp']]]
            blk_arr[:,off1-1]=numpy.transpose(indc)
            temp=X[:,indc]
            temp = temp[:,:,0]
            DC[:,off1-1]=numpy.mean(temp)
            #print 'DC: '+str(DC.shape)
            bigbadindex = numpy.arange((off1-1) * par['nlsp'],off1 * par['nlsp'])
            #print bigbadindex.shape, off1, par['nlsp']
            #print DC[:,off1].shape, temp.shape, bigbadindex.shape
            #print nDCnlX[:,bigbadindex].shape
            tempqwe = numpy.transpose(numpy.transpose(temp) - (DC[:,off1-1]))
            #print tempqwe.shape
            nDCnlX[:,bigbadindex] = tempqwe#numpy.transpose(numpy.transpose(temp) - (DC[:,off1]))
    return nDCnlX,blk_arr,DC,par
