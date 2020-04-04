import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
import time
import warnings
import subprocess
np.set_printoptions(threshold=np.nan)
from scipy import stats
from lib.tools3d import *
from lib.tools2d import *
from lib.fileio import *
from lib.est3D import *

def main(inputs, inds, XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, n, nlevels, nparams):

    # ----------------------------------------------------------------------
    #  Read in one input nifti to get size, affines, etc.
    # ----------------------------------------------------------------------
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = blmm_load(nifti_path)

    NIFTIsize = nifti.shape


    # ----------------------------------------------------------------------
    # Input variables
    # ----------------------------------------------------------------------

    # Output directory
    OutDir = inputs['outdir']

    # Convergence tolerance
    if "tol" in inputs:
        tol=inputs['tol']
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
    qu = np.sum(nparams*(nparams+1)//2) # (Number of unique random effects)


    # REML is just a backdoor option at the moment as it isn't that useful
    # in the large n setting. For now we just set it to false.
    REML = False

    # ----------------------------------------------------------------------
    # Parameter estimation
    # ----------------------------------------------------------------------  

    if method=='pSFS': # Recommended, default method
        paramVec = pSFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6, n, reml=REML)
    
    if method=='FS': 
        paramVec = FS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6, n, reml=REML)

    if method=='SFS': 
        paramVec = SFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6, n, reml=REML)

    if method=='pFS': 
        paramVec = pFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6, n, reml=REML)

    # ----------------------------------------------------------------------
    # Parameter outputting
    # ----------------------------------------------------------------------    

    # Dimension of beta volume
    dimBeta = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],p)

    # Dimension of D volume
    dimD = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],qu)

    # Get the indices in the paramvector corresponding to D matrices
    IndsDk = np.int32(np.cumsum(nparams*(nparams+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # Assign betas
    beta = paramVec[:, 0:p]
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_beta.nii'), beta, inds,volc=None,dim=dimBeta,aff=nifti.affine,hdr=nifti.header)        
    
    sigma2 = paramVec[:,p:(p+1),:]
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_sigma2.nii'), sigma2, inds,volc=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

    vechD = paramVec[:,(p+1):,:].reshape((v,qu))
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_D.nii'), vechD, inds,volc=None,dim=dimD,aff=nifti.affine,hdr=nifti.header) 

    Ddict = dict()
    # D as a dictionary
    for k in np.arange(len(nparams)):

        Ddict[k] = vech2mat3D(paramVec[:,IndsDk[k]:IndsDk[k+1],:])
      
    # Full version of D
    D = getDfromDict3D(Ddict, nparams, nlevels)

    return(beta, sigma2, D)