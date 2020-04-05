import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import sys
import os
import glob
import shutil
import yaml
np.set_printoptions(threshold=np.nan)
from lib.npMatrix3d import *
from lib.npMatrix2d import *
from lib.fileio import *

# --------------------------------------------------------------------------
# Author: Tom Maullin (04/04/2020)

def main(inputs, nparams, nlevels, inds, beta, D, sigma2, n, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ):

    # ----------------------------------------------------------------------
    #  Read in one input nifti to get size, affines, etc.
    # ----------------------------------------------------------------------
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = loadFile(nifti_path)

    NIFTIsize = nifti.shape

    # ----------------------------------------------------------------------
    # Input variables
    # ----------------------------------------------------------------------

    # Output directory
    OutDir = inputs['outdir']

    # Value to replace -inf with in -log10(p) maps.
    if "minlog" in inputs:
        minlog=inputs['minlog']
    else:
        minlog=-323.3062153431158

    # ----------------------------------------------------------------------
    # Preliminary useful variables
    # ---------------------------------------------------------------------- 

    # Scalar quantities
    v = np.prod(inds.shape) # (Number of voxels we are looking at)
    p = XtX.shape[1] # (Number of Fixed Effects parameters)
    q = np.sum(nparams*nlevels) # (Total number of random effects)
    qu = np.sum(nparams*(nparams+1)//2) # (Number of unique random effects)
    c = len(inputs['contrasts']) # (Number of contrasts)

    # Reshape n if necessary
    if isinstance(n,np.ndarray):
        # Check first that n isn't a single value
        if np.prod(n.shape)>1:
            # Reshape
            n = n.reshape(v) # (Number of inputs)

    # Miscellaneous matrix variables
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr3D(YtX, YtY, XtX, beta)

    # REML (currently only exists as a backdoor option as is not much 
    # practical use in the high n setting)
    REML = False

    # ----------------------------------------------------------------------
    # Calculate log-likelihood
    # ---------------------------------------------------------------------- 

    # Output log likelihood
    llh = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD, D, REML, XtX, XtZ, ZtX) - (0.5*(n)*np.log(2*np.pi))
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_llh.nii'), llh, inds,volc=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

    # ----------------------------------------------------------------------
    # Calculate residual mean squares = e'e/(n - p)
    #
    # Note: In the mixed model resms is different to our sigma2 estimate as:
    #
    #  - resms = e'e/(n-p)
    #  - sigma2 = e'V^(-1)e/n for "Simplified methods" or has no closed form
    #             expression for more general methods
    #
    # ----------------------------------------------------------------------
    resms = get_resms3D(YtX, YtY, XtX, beta,n,p).reshape(v)
    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_resms.nii'), resms, inds,volc=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)

    # ----------------------------------------------------------------------
    # Calculate beta covariance maps (Optionally output)
    # ----------------------------------------------------------------------

    if "OutputCovB" in inputs:
        OutputCovB = inputs["OutputCovB"]
    else:
        OutputCovB = True

    if OutputCovB:

        # Dimensoon of cov(beta) NIFTI
        dimCov = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],p**2)

        # Work out cov(beta)
        covB = get_covB3D(XtX, XtZ, DinvIplusZtZD, sigma2).reshape(v, p**2)
        addBlockToNifti(os.path.join(OutDir, 'blmm_vox_cov.nii'), covB, inds,volc=None,dim=dimCov,aff=nifti.affine,hdr=nifti.header)
        del covB

    # ----------------------------------------------------------------------
    # Calculate COPEs, statistic maps and covariance maps.
    # ----------------------------------------------------------------------
    # Record how many T contrasts and F contrasts we have seen
    nt = 0
    nf = 0

    # Count the number of T contrasts and F contrasts in the inputs
    for i in range(0,c):

        # Read in contrast vector
        L = str2vec(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        L = np.array(L)

        if L.ndim == 1:
            nt = nt + 1
        else:
            nf = nf + 1

    # Current number for contrast (T and F)
    current_nt = 0
    current_nf = 0

    # Loop through contrasts
    for i in range(0,c):

        # Read in contrast vector
        L = str2vec(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        L = np.array(L)
    
        # Work out if it is a T or an F contrast NTS: FIX THIS
        if L.ndim == 1:
            statType='T'
            L = L.reshape([1,L.shape[0]])
        else:
            statType='F'

        # ------------------------------------------------------------------
        # T contrasts
        # ------------------------------------------------------------------
        if statType == 'T':

            # Work out the dimension of the T-stat-related volumes
            dimT = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],nt)

            # Work out L\beta
            Lbeta = L @ beta
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_con.nii'), Lbeta, inds,volc=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

            # Work out s.e.(L\beta)
            seLB = np.sqrt(get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2).reshape(v))
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conSE.nii'), seLB, inds,volc=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)


            # Calculate sattherwaite estimate of the degrees of freedom of this statistic
            swdfc = get_swdf_T3D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nparams).reshape(v)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conT_swedf.nii'), swdfc, inds,volc=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

            # Obtain and output T statistic
            Tc = get_T3D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2).reshape(v)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conT.nii'), Tc, inds,volc=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

            # Obatin and output p-values
            pc = T2P3D(Tc,swdfc,minlog)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conTlp.nii'), pc, inds,volc=current_nt,dim=dimT,aff=nifti.affine,hdr=nifti.header)

            # Record that we have seen another T contrast
            current_nt = current_nt + 1

        # ------------------------------------------------------------------
        # F contrasts
        # ------------------------------------------------------------------
        if statType == 'F':

            # Work out the dimension of the F-stat-related volumes
            dimF = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],nf)

            # Calculate sattherthwaite degrees of freedom for the inner.
            swdfc = get_swdf_F3D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nparams).reshape(v)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conF_swedf.nii'), swdfc, inds,volc=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

            # Calculate F statistic.
            Fc=get_F3D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2).reshape(v)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conF.nii'), Fc, inds,volc=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

            # Work out p for this contrast
            pc = F2P3D(Fc, L, swdfc, minlog).reshape(v)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conFlp.nii'), pc, inds,volc=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

            # Calculate partial R2 masked for ring.
            R2 = get_R23D(L, Fc, swdfc).reshape(v)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conR2.nii'), R2, inds,volc=current_nf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

            # Record that we have seen another F contrast
            current_nf = current_nf + 1