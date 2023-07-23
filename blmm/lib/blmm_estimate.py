import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import os
import time
import sys
np.set_printoptions(threshold=sys.maxsize)
from blmm.src.npMatrix3d import *
from blmm.src.npMatrix2d import *
from blmm.src.fileio import *
from blmm.src.est3d import *


# ====================================================================================
#
# This file is the fourth stage of the BLMM pipeline. Here, the parameters beta, 
# sigma2 and D are estimated from the product matrices. By default the estimation 
# method is set to pSFS (which has been observed to be the quickest), but all 3D 
# Fisher Scoring methods have been included as options for completeness. The parameter
# estimates are also output as NIFTI images here.
#
# ------------------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited: 04/04/2020)
#
# ------------------------------------------------------------------------------------
#
# The code takes the following inputs:
#
# ------------------------------------------------------------------------------------
#
#  - `inputs`: The contents of the `inputs.yml` file, loaded using the `yaml` python 
#              package.
#  - `inds`: The (flattened) indices of the voxels we wish to perform parameter
#            estimation for.
#  - `XtX`: X transpose multiplied by X (can be spatially varying or non-spatially 
#           varying). 
#  - `XtY`: X transpose multiplied by Y (spatially varying).
#  - `XtZ`: X transpose multiplied by Z (can be spatially varying or non-spatially 
#           varying).
#  - `YtX`: Y transpose multiplied by X (spatially varying).
#  - `YtY`: Y transpose multiplied by Y (spatially varying).
#  - `YtZ`: Y transpose multiplied by Z (spatially varying).
#  - `ZtX`: Z transpose multiplied by X (can be spatially varying or non-spatially 
#           varying).
#  - `ZtY`: Z transpose multiplied by Y (spatially varying).
#  - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non-spatially 
#           varying). If we are looking at a one random factor one random effect 
#           design the variable ZtZ only holds the diagonal elements of the matrix
#           Z'Z.
#  - `n`: The number of observations (can be spatially varying or non-spatially 
#         varying). 
#  - `nlevels`: A vector containing the number of levels for each factor, e.g. 
#               `nlevels=[3,4]` would mean the first factor has 3 levels and the
#               second factor has 4 levels.
#  - `nraneffs`: A vector containing the number of random effects for each
#                factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#                random effects and the second factor has 1 random effect.
#
# ------------------------------------------------------------------------------------
#
# And returns as outputs:
#
# ------------------------------------------------------------------------------------
#
# - `beta`: The fixed effects parameter estimates for each voxel.
# - `sigma2`: The fixed effects variance estimate for each voxel.
# - `D`: The random effects covariance matrix estimate for each voxel.
#
# ====================================================================================
def estimate(inputs, inds, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, n, nlevels, nraneffs):

    # ----------------------------------------------------------------------
    #  Read in one input nifti to get size, affines, etc.
    # ----------------------------------------------------------------------
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = loadFile(nifti_path)

    # Work out the dimensions of the NIFTI images
    NIFTIsize = nifti.shape


    # ----------------------------------------------------------------------
    # Input variables
    # ----------------------------------------------------------------------

    # Output directory
    OutDir = inputs['outdir']

    # Convergence tolerance
    if "tol" in inputs:
        tol=eval(inputs['tol'])
    else:
        tol=1e-6

    # Estimation method
    if "method" in inputs:
        method=inputs['method']
    else:
        method='pSFS'

    # ----------------------------------------------------------------------
    # Preliminary useful variables
    # ---------------------------------------------------------------------- 

    # Scalar quantities
    v = np.prod(inds.shape) # (Number of voxels we are looking at)
    p = XtX.shape[1] # (Number of Fixed Effects parameters)
    qu = np.sum(nraneffs*(nraneffs+1)//2) # (Number of unique random effects)


    # REML is just a backdoor option at the moment. For now we just set it
    # to false.
    REML = True

    # ----------------------------------------------------------------------
    # Parameter estimation
    # ----------------------------------------------------------------------  
    # If running simulations we record the time used for parameter estimation
    if 'sim' in inputs:
        if inputs['sim']:
            t1 = time.time()

    if 'maxnit' in inputs:
        maxnit = int(inputs['maxnit'])
    else:
        maxnit = 10000

    if method=='pSFS': # Recommended, default method
        paramVec = pSFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, reml=REML, maxnit=maxnit)
    
    if method=='FS': 
        paramVec = FS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n)

    if method=='SFS': 
        paramVec = SFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n)

    if method=='pFS': 
        paramVec = pFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n)


    # If running simulations we record the time used for parameter estimation
    if 'sim' in inputs:
        if inputs['sim']:
            t2 = time.time()

            # Output an "average estimation time nifti"
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_times.nii'), np.ones((v,1))*(t2-t1)/v, inds,volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

    # ----------------------------------------------------------------------
    # Parameter outputting
    # ----------------------------------------------------------------------    

    # Dimension of beta volume
    dimBeta = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],p)

    # Dimension of D volume
    dimD = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],qu)

    # Get the indices in the paramvector corresponding to D matrices
    IndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # Output beta estimate
    beta = paramVec[:, 0:p]
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_beta.nii'), beta, inds,volInd=None,dim=dimBeta,aff=nifti.affine,hdr=nifti.header)        
    
    # Output sigma2 estimate
    sigma2 = paramVec[:,p:(p+1),:]
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_sigma2.nii'), sigma2, inds,volInd=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

    # Output unique D elements (i.e. [vech(D_1),...vech(D_r)])
    vechD = paramVec[:,(p+1):,:].reshape((v,qu))
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_D.nii'), vechD, inds,volInd=None,dim=dimD,aff=nifti.affine,hdr=nifti.header) 

    # Reconstruct D
    Ddict = dict()
    # D as a dictionary
    for k in np.arange(len(nraneffs)):

        Ddict[k] = vech2mat3D(paramVec[:,IndsDk[k]:IndsDk[k+1],:])
      
    # Full version of D
    D = getDfromDict3D(Ddict, nraneffs, nlevels)

    return(beta, sigma2, D)