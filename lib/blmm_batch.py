import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import shutil
import yaml
import time
np.set_printoptions(threshold=np.nan)
from lib.blmm_eval import blmm_eval
from lib.blmm_load import blmm_load
import scipy.sparse
import pandas as pd

def main(*args):

    # Change to blm directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    

    batchNo = args[0]

    if len(args)==1 or (not args[1]):
        # Load in inputs
        with open(os.path.join(os.getcwd(),'..','blmm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        if type(args[1]) is str:
            # In this case inputs file is first argument
            with open(os.path.join(args[1]), 'r') as stream:
                inputs = yaml.load(stream)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[1]

    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    OutDir = inputs['outdir']

    # Get number of ffx parameters
    c1 = blmm_eval(inputs['contrasts'][0]['c' + str(1)]['vector'])
    c1 = np.array(c1)
    n_p = c1.shape[0]
    del c1

    # Y volumes
    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load in one nifti to check NIFTI size
    try:
        Y0 = blmm_load(Y_files[0])
    except Exception as error:
        raise ValueError('The NIFTI "' + Y_files[0] + '"does not exist')

    d0 = Y0.get_data()

    # Get the maximum memory a NIFTI could take in storage. 
    NIFTIsize = sys.getsizeof(np.zeros(d0.shape,dtype='uint64'))

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use.
    blksize = int(np.floor(MAXMEM/8/NIFTIsize/n_p));

    # Reduce X to X for this block.
    X = blmm_load(inputs['X'])
    print(X.shape)
    X = X[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]
    

    # Number of random effects factors.
    n_f = len(inputs['Z'])

    # Read in each factor
    for i in range(0,n_f):

        Zi_factor = blmm_load(inputs['Z'][i]['f' + str(i+1)]['factor'])
        Zi_design = blmm_load(inputs['Z'][i]['f' + str(i+1)]['design'])

        # Number of levels for factor i
        l_i = np.amax(Zi_factor)

        # Number of parameters for factor i
        q_i = Zi_design.shape[1]

        print('l')
        print(l_i)
        print('q')
        print(q_i)

        print("Z shapes (full)")
        print(Zi_design.shape)
        print(Zi_factor.shape)

        # One hot encode the factor vector
        Zi_factor = pd.get_dummies(pd.DataFrame(Zi_factor)[0]).values

        # Reduce to block.
        Zi_design = Zi_design[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]
        Zi_factor = Zi_factor[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]

        print("Z shapes (block)")
        print(Zi_design.shape)
        print(Zi_factor.shape)


        # Repeat Zi_factor for each parameter
        Zi = np.repeat(Zi_factor, q_i,axis=1).astype(np.float64)

        # Fill the one values with the design
        Zi[Zi==1]=Zi_design.reshape(Zi[Zi==1].shape)

        print('Z shape')
        print(Zi.shape)

        # Concatenate Z's Horizontally
        if i == 0:

            Z = Zi

        else:

            Z = np.hstack((Z,Zi))

    print('XZ', X.shape, Z.shape)

    # Mask volumes (if they are given)
    if 'data_mask_files' in inputs:

        with open(inputs['data_mask_files']) as a:

            M_files = []
            i = 0
            for line in a.readlines():

                M_files.append(line.replace('\n', ''))

        # If we have a mask for each Y, reduce the list to just for this block
        if len(M_files) == len(Y_files):

            # In this case we have a mask per Y volume
            M_files = M_files[(blksize*(batchNo-1)):min((blksize*batchNo),len(M_files))]

        else:

            if len(M_files) > len(Y_files):

                raise ValueError('Too many data_masks specified!')

            else:

                raise ValueError('Too few data_masks specified!')

    # Otherwise we have no masks
    else:

        # There is not a mask for each Y as there are no masks at all!
        M_files = []

    # Mask threshold for Y (if given)
    if 'data_mask_thresh' in inputs:
        M_t = float(inputs['data_mask_thresh'])
    else:
        M_t = None

    # Reduce Y_files to only Y files for this block
    Y_files = Y_files[(blksize*(batchNo-1)):min((blksize*batchNo),len(Y_files))]

    print('Y_files 0')
    print(Y_files[0])
    
    # Verify input
    verifyInput(Y_files, M_files, Y0)

    # Obtain Y, mask for Y and nmap. This mask is just for voxels
    # with no studies present.
    Y, Mask, nmap = obtainY(Y_files, M_files, M_t)

    print(Y.shape)

    # Work out voxel specific designs
    MX = blkMX(X, Y)
    print('MXY ran')
    MZ = blkMX(Z, Y) # MIGHT NEED TO THINK ABOUT SPARSITY HERE LATER
    print('MZY ran')
    
    # Get X transpose Y, Z transpose Y and Y transpose Y.
    XtY = blkXtY(X, Y, Mask)
    print('XtY ran')
    YtY = blkYtY(Y, Mask)
    print('YtY ran')
    ZtY = blkXtY(Z, Y, Mask) # REVISIT ONCE SPARSED
    print('ZtY ran')

    # In a spatially varying design XtX has dimensions n_voxels
    # by n_parameters by n_parameters. We reshape to n_voxels by
    # n_parameters^2 so that we can save as a csv.
    XtX_m = blkXtX(MX)
    XtX_m = XtX_m.reshape([XtX_m.shape[0], XtX_m.shape[1]*XtX_m.shape[2]])

    # We then need to unmask XtX as we now are saving XtX.
    XtX = np.zeros([Mask.shape[0],XtX_m.shape[1]])
    XtX[np.flatnonzero(Mask),:] = XtX_m[:]

    # In a spatially varying design ZtX has dimensions n_voxels
    # by n_q by n_parameters. We reshape to n_voxels by
    # n_parameters^2 so that we can save as a csv.
    ZtX_m = blkZtX(MZ, MX)
    ZtX_m = XtX_m.reshape([ZtX_m.shape[0], ZtX_m.shape[1]*ZtX_m.shape[2]])

    # We then need to unmask XtX as we now are saving XtX.
    ZtX = np.zeros([Mask.shape[0],ZtX_m.shape[1]])
    ZtX[np.flatnonzero(Mask),:] = ZtX_m[:]

    # ======================================================================
    # NEED TO THINK ABOUT SPARSE HERE
    # ======================================================================
    #
    # In a spatially varying design ZtZ has dimensions n_voxels
    # by n_q by n_q. We reshape to n_voxels by n_q^2 so that we
    # can save as a csv.
    ZtZ_m = blkXtX(MZ)
    ZtZ_m = ZtZ_m.reshape([ZtZ_m.shape[0], ZtZ_m.shape[1]*ZtZ_m.shape[2]])

    # We then need to unmask XtX as we now are saving XtX.
    ZtZ = np.zeros([Mask.shape[0],ZtZ_m.shape[1]])
    ZtZ[np.flatnonzero(Mask),:] = ZtZ_m[:]

    # ======================================================================

    # Pandas reads and writes files much more quickly with nrows <<
    # number of columns
    XtY = XtY.transpose()
    ZtY = ZtY.transpose()

    print('got here')

    if (len(args)==1) or (type(args[1]) is str):

        print('also got here')
        # Record product matrices 
        np.save(os.path.join(OutDir,"tmp","XtX" + str(batchNo)), 
                   XtX)
        np.save(os.path.join(OutDir,"tmp","XtY" + str(batchNo)), 
                   XtY) 
        np.save(os.path.join(OutDir,"tmp","YtY" + str(batchNo)), 
                   YtY) 
        np.save(os.path.join(OutDir,"tmp","ZtY" + str(batchNo)), 
                   ZtY) 
        np.save(os.path.join(OutDir,"tmp","ZtX" + str(batchNo)), 
                   ZtX) 
        np.save(os.path.join(OutDir,"tmp","ZtZ" + str(batchNo)), 
                   ZtZ) 

        # Get map of number of scans at voxel.
        nmap = nib.Nifti1Image(nmap,
                               Y0.affine,
                               header=Y0.header)
        nib.save(nmap, os.path.join(OutDir,'tmp',
                        'blmm_vox_n_batch'+ str(batchNo) + '.nii'))
    else:
        # Return XtX, XtY, YtY, nB
        return(XtX, XtY, YtY, nmap)
    w.resetwarnings()

def verifyInput(Y_files, M_files, Y0):

    # Obtain information about zero-th scan
    d0 = Y0.get_data()
    Y0aff = Y0.affine

    # Initial checks for NIFTI compatability for Y.
    for i in range(0, len(Y_files)):

        Y_file = Y_files[i]

        try:
            Y = blmm_load(Y_file)
        except Exception as error:
            raise ValueError('The NIFTI "' + Y_file + '"does not exist')

        # Check NIFTI images have the same dimensions.
        if not np.array_equal(Y0.shape, Y.shape):
            raise ValueError('Input NIFTI "' + Y_file + '" has ' +
                             'different dimensions to "' +
                             Y0 + '"')

        # Check NIFTI images are in the same space.
        if not np.array_equal(Y.affine, Y0aff):
            raise ValueError('Input NIFTI "' + Y_file + '" has a ' +
                             'different affine transformation to "' +
                             Y0 + '"')

    # Initial checks for NIFTI compatability for M.
    if M_files is not None:
        for i in range(0, len(M_files)):

            M_file = M_files[i]

            try:
                M = blmm_load(M_file)
            except Exception as error:
                raise ValueError('The NIFTI "' + M_file + '"does not exist')

            # Check NIFTI images have the same dimensions.
            if not np.array_equal(Y0.shape, M.shape):
                raise ValueError('Input NIFTI "' + M_file + '" has ' +
                                 'different dimensions to "' +
                                 Y0 + '"')

            # Check NIFTI images are in the same space.
            if not np.array_equal(M.affine, Y0aff):
                raise ValueError('Input NIFTI "' + M_file + '" has a ' +
                                 'different affine transformation to "' +
                                 Y0 + '"')

def blkMX(X,Y):

    # Work out the mask.
    M = (Y!=0)

    # Get M in a form where each voxel's mask is mutliplied
    # by X
    M = M.transpose().reshape([M.shape[1], 1, M.shape[0]])
    Xt=X.transpose()

    # Obtain design for each voxel
    MXt = np.multiply(M, Xt)
    MX = MXt.transpose(0,2,1)

    return MX

def obtainY(Y_files, M_files, M_t):

    # Load in one nifti to check NIFTI size
    Y0 = blmm_load(Y_files[0])
    d = Y0.get_data()
    
    # Get number of voxels.
    nvox = np.prod(d.shape)

    # Number of scans in block
    nscan = len(Y_files)

    # Count number of scans contributing to voxels
    nmap = np.zeros(d.shape)

    # Read in Y
    Y = np.zeros([nscan, nvox])
    for i in range(0, len(Y_files)):

        # Read in each individual NIFTI.
        Y_indiv = blmm_load(Y_files[i])

        # Mask Y if necesart
        if M_files:
        
            # Apply mask
            M_indiv = blmm_load(M_files[i]).get_data()
            d = np.multiply(
                Y_indiv.get_data(),
                M_indiv)
        else: 
            #Just load in Y
            d = Y_indiv.get_data()

        # If theres an initial threshold for the data apply it.
        if M_t is not None:
            d[d<M_t]=0

        # NaN check
        d = np.nan_to_num(d)

        # Count number of scans at each voxel
        nmap = nmap + 1*(np.nan_to_num(d)!=0)

        # Constructing Y matrix
        Y[i, :] = d.reshape([1, nvox])
    
    Mask = np.zeros([nvox])
    Mask[np.where(np.count_nonzero(Y, axis=0)>0)[0]] = 1
    
    Y = Y[:, np.where(np.count_nonzero(Y, axis=0)>0)[0]]

    return Y, Mask, nmap

# Note: this techniqcally calculates sum(Y.Y) for each voxel,
# not Y transpose Y for all voxels
def blkYtY(Y, Mask):

    # Read in number of scans and voxels.
    nscan = Y.shape[0]
    nvox = Y.shape[1]
    
    # Reshape Y
    Y_rs = Y.transpose().reshape(nvox, nscan, 1)
    Yt_rs = Y.transpose().reshape(nvox, 1, nscan)
    del Y

    # Calculate Y transpose Y.
    YtY_m = np.matmul(Yt_rs,Y_rs).reshape([nvox, 1])

    # Unmask YtY
    YtY = np.zeros([Mask.shape[0], 1])
    YtY[np.flatnonzero(Mask),:] = YtY_m[:]

    return YtY


def blkXtY(X, Y, Mask):
    
    # Calculate X transpose Y (Masked)
    XtY_m = np.asarray(
                np.dot(np.transpose(X), Y))

    # Check the dimensions haven't been reduced
    # (numpy will lower the dimension of the 
    # array if the length in one dimension is
    # one)
    if np.ndim(XtY_m) == 0:
        XtY_m = np.array([[XtY_m]])
    elif np.ndim(XtY_m) == 1:
        XtY_m = np.array([XtY_m])

    # Unmask XtY
    XtY = np.zeros([XtY_m.shape[0], Mask.shape[0]])
    XtY[:,np.flatnonzero(Mask)] = XtY_m[:]

    return XtY


def blkXtX(X):

    if np.ndim(X) == 3:

        Xt = X.transpose((0, 2, 1))
        XtX = np.matmul(Xt, X)

    else:

        # Calculate XtX
        XtX = np.asarray(
                    np.dot(np.transpose(X), X))

        # Check the dimensions haven't been reduced
        # (numpy will lower the dimension of the 
        # array if the length in one dimension is
        # one)
        if np.ndim(XtX) == 0:
            XtX = np.array([XtX])
        elif np.ndim(XtX) == 1:
            XtX = np.array([XtX])

    return XtX


def blkZtX(Z,X):

    if np.ndim(Z) == 3:

        Zt = Z.transpose((0, 2, 1))
        ZtX = np.matmul(Zt, X)

    else:

        # Calculate XtX
        ZtX = np.asarray(
                    np.dot(np.transpose(Z), X))

        # Check the dimensions haven't been reduced
        # (numpy will lower the dimension of the 
        # array if the length in one dimension is
        # one)
        if np.ndim(ZtX) == 0:
            ZtX = np.array([ZtX])
        elif np.ndim(ZtX) == 1:
            ZtX = np.array([ZtX])

    return ZtX



if __name__ == "__main__":
    main()
