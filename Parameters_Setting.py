import scipy.io as sio
import numpy

# Translated to Python by Duncan Sommer

def Parameters_Setting(nSig):
    par = {}

    # the step of two neighbor patches
    par['step'] = 3
    # the iteration number
    par['IteNum'] = 4
    par['nSig'] = nSig/255

    if nSig<=10:
        filename = './model/PG_GMM_6x6_win15_nlsp10_delta0.002_cls65.mat'
        par['c1'] = 0.33*2*numpy.sqrt(2)
        par['delta'] = 0.1
        par['eta'] = 0.79
 
    elif nSig<=20:
        filename = './model/PG_GMM_6x6_win15_nlsp10_delta0.002_cls65.mat'
        par['c1'] = 0.29*2*numpy.sqrt(2)
        par['delta'] = 0.09
        par['eta'] = 0.73
 
    elif nSig<=30:
        filename = './model/PG_GMM_7x7_win15_nlsp10_delta0.002_cls33.mat'
        par['c1'] = 0.19*2*numpy.sqrt(2)
        par['delta'] = 0.08
        par['eta'] = 0.89
 
    elif nSig<=40:
        filename = './model/PG_GMM_8x8_win15_nlsp10_delta0.002_cls33.mat'
        par['c1'] = 0.15*2*numpy.sqrt(2)
        par['delta'] = 0.07
        par['eta'] = 0.98
 
    elif nSig<=50:
        filename = './model/PG_GMM_8x8_win15_nlsp10_delta0.002_cls33.mat'
        par['c1'] = 0.12*2*numpy.sqrt(2)
        par['delta'] = 0.06
        par['eta'] = 1.05
 
    elif nSig<=75:
        filename = './model/PG_GMM_9x9_win15_nlsp10_delta0.002_cls33.mat'
        par['c1'] = 0.09*2*numpy.sqrt(2)
        par['delta'] = 0.05  
        par['eta'] = 1.15
 
    else:
        filename = './model/PG_GMM_9x9_win15_nlsp10_delta0.002_cls33.mat'
        par['c1'] = 0.06*2*numpy.sqrt(2)
        par['delta'] = 0.05
        par['eta'] = 1.30


    # Load the saved model data into a local dictionary
    load_dict = sio.loadmat(filename, squeeze_me=True, verify_compressed_data_integrity = True)

    # Extract the model itself (BIG)
    model = load_dict['model']
    # Some variables didn't get loaded properly
    #model['n_models'] = load_dict['cls_num']


    #tmp = model['covs'].item()
    #del model['covs']
    #model['covs'] = tmp

	# Extract necessary parameter data

    par['ps'] = load_dict['ps']        # patch size
    par['nlsp'] = load_dict['nlsp']    # number of non-local patches
    par['Win'] = load_dict['win']      # size of window around the patch
 
 
    # dictionary and regularization parameter

    # I'm just making par.D a list
    # Don't convert to singles from float
    par['D'] = []
    S_len = len(load_dict['GMM_S'])
    for i in range(len(load_dict['GMM_D'][0])):
        par['D'].append( numpy.reshape(load_dict['GMM_D'][:,i], (S_len, S_len), order='F') )

    # Original version of above loop:
    # for i = 1:size(GMM_D,2)
    #     par.D(:,:,i) = reshape(single(GMM_D(:, i)), size(GMM_S,1), size(GMM_S,1));
    # end


    # Let's try leaving this array as doubles too. It shouldn't hurt, right?
    #par.S = single(GMM_S)

    return (par, model)


# load_dict = sio.loadmat('./model/PG_GMM_8x8_win15_nlsp10_delta0.002_cls33.mat', squeeze_me=True)
