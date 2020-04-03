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
from lib.blmm_eval import blmm_eval
np.set_printoptions(threshold=np.nan)
from scipy import stats
from lib.blmm_load import blmm_load
from lib.tools3d import *
from lib.tools2d import *
from lib.pSFS import pSFS

# Developer notes:
# --------------------------------------------------------------------------
# In the following code I have used the following subscripts to indicate:
#
# _r - This means this is an array of values corresponding to voxels which
#      are present in between k and n_s-1 studies (inclusive), where k is
#      decided by the user specified thresholds. These voxels will typically
#      be on the edge of the brain and look like a "ring" around the brain,
#      hence "_r" for ring.
# 
# _i - This means that this is an array of values corresponding to voxels 
#      which are present in all n_s studies. These will usually look like
#      a smaller mask place inside the whole study mask. Hence "_i" for 
#      inner.
#
# _sv - This means this variable is spatially varying (There is a reading
#       per voxel). 
#
# --------------------------------------------------------------------------
# Author: Tom Maullin (04/02/2019)

def main(*args):

    # ----------------------------------------------------------------------
    # Check inputs
    # ----------------------------------------------------------------------
    if len(args)==0 or (not args[0]):
        # Load in inputs
        with open(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..',
                    'blmm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream,Loader=yaml.FullLoader)
    else:
        if type(args[0]) is str:
            # In this case inputs file is first argument
            with open(os.path.join(args[0]), 'r') as stream:
                inputs = yaml.load(stream,Loader=yaml.FullLoader)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[0]

    # ----------------------------------------------------------------------
    # Read basic inputs
    # ----------------------------------------------------------------------
    OutDir = inputs['outdir']

    # Random factor variables.
    rfxmats = inputs['Z']

    # Number of random effects
    r = len(rfxmats)

    # Number of variables in each factor, q
    nparams = []

    # Number of levels for each factor, l
    nlevels = []

    for k in range(r):

        rfxdes = blmm_load(rfxmats[k]['f' + str(k+1)]['design'])
        rfxfac = blmm_load(rfxmats[k]['f' + str(k+1)]['factor'])

        nparams = nparams + [rfxdes.shape[1]]
        nlevels = nlevels + [len(np.unique(rfxfac))]

    # Get number of rfx params
    nparams = np.array(nparams)
    nlevels = np.array(nlevels)
    n_q = np.sum(nparams*nlevels)

    # Get number of unique rfx params
    n_q_u = np.sum(nparams*(nparams+1)//2)
    
    # Get number of parameters
    c1 = blmm_eval(inputs['contrasts'][0]['c' + str(1)]['vector'])
    c1 = np.array(c1)
    n_p = c1.shape[0]
    del c1
    
    # Read in the nifti size and work out number of voxels.
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = blmm_load(nifti_path)

    NIFTIsize = nifti.shape
    n_v = int(np.prod(NIFTIsize))

    # ----------------------------------------------------------------------
    # Remove any files from the previous runs
    #
    # Note: This is important as if we are outputting blocks to files we
    # want to be sure none of the previous results are lingering around 
    # anywhere.
    # ----------------------------------------------------------------------

    files = ['blmm_vox_n.nii', 'blmm_vox_mask.nii', 'blmm_vox_edf.nii', 'blmm_vox_beta.nii',
             'blmm_vox_llh.nii', 'blmm_vox_sigma2.nii', 'blmm_vox_D.nii', 'blmm_vox_resms.nii',
             'blmm_vox_cov.nii', 'blmm_vox_conT_swedf.nii', 'blmm_vox_conT.nii', 'blmm_vox_conTlp.nii',
             'blmm_vox_conSE.nii', 'blmm_vox_con.nii', 'blmm_vox_conF.nii', 'blmm_vox_conF_swedf.nii',
             'blmm_vox_conFlp.nii', 'blmm_vox_conR2.nii']

    for file in files:

        if os.path.exists(os.path.join(OutDir, file)):

            os.remove(os.path.join(OutDir, file))

    # ----------------------------------------------------------------------
    # Get n_s (number of subjects) and n_s_sv (spatially varying number of
    # subjects)
    # ----------------------------------------------------------------------

    # Work out number of batchs
    n_b = len(glob.glob(os.path.join(OutDir,"tmp","blmm_vox_n_batch*")))

    if (len(args)==0) or (type(args[0]) is str):

        # Read in n_s (spatially varying)
        nmapb  = blmm_load(os.path.join(OutDir,"tmp", "blmm_vox_n_batch1.nii"))
        n_s_sv = nmapb.get_data()# Read in uniqueness Mask file

        # Remove files, don't need them anymore
        os.remove(os.path.join(OutDir,"tmp","blmm_vox_n_batch1.nii"))

        # Cycle through batches and add together n.
        for batchNo in range(2,(n_b+1)):
            
            # Obtain the full nmap.
            n_s_sv = n_s_sv + blmm_load(os.path.join(OutDir,"tmp", 
                "blmm_vox_n_batch" + str(batchNo) + ".nii")).get_data()
            
            # Remove file, don't need it anymore
            os.remove(os.path.join(OutDir, "tmp", "blmm_vox_n_batch" + str(batchNo) + ".nii"))

    else:
        # Read in n_s_sv.
        n_s_sv = args[4]

    # Save nmap
    nmap = nib.Nifti1Image(n_s_sv,
                           nifti.affine,
                           header=nifti.header)
    nib.save(nmap, os.path.join(OutDir,'blmm_vox_n.nii'))
    n_s_sv = n_s_sv.reshape(n_v, 1)
    del nmap

    # Get ns.
    X = blmm_load(inputs['X'])
    n_s = X.shape[0]

    # ----------------------------------------------------------------------
    # Create Mask
    # ----------------------------------------------------------------------

    Mask = np.ones([n_v, 1])

    # Check for user specified missingness thresholds.
    if 'Missingness' in inputs:

        # Apply user specified missingness thresholding.
        if ("MinPercent" in inputs["Missingness"]) or ("minpercent" in inputs["Missingness"]):

            # Read in relative threshold
            if "MinPercent" in inputs["Missingness"]:
                rmThresh = inputs["Missingness"]["MinPercent"]
            else:
                rmThresh = inputs["Missingness"]["minpercent"]

            # If it's a percentage it will be a string and must be converted.
            rmThresh = str(rmThresh)
            if "%" in rmThresh:
                rmThresh = float(rmThresh.replace("%", ""))/100
            else:
                rmThresh = float(rmThresh)

            # Check the Relative threshold is between 0 and 1.
            if (rmThresh < 0) or (rmThresh > 1):
                raise ValueError('Minumum percentage missingness threshold is out of range: ' +
                                 '0 < ' + str(rmThresh) + ' < 1 violation')

            # Mask based on threshold.
            Mask[n_s_sv<rmThresh*n_s]=0

        if ("MinN" in inputs["Missingness"]) or ("minn" in inputs["Missingness"]):

            # Read in relative threshold
            if "minn" in inputs["Missingness"]:
                amThresh = inputs["Missingness"]["minn"]
            else:
                amThresh = inputs["Missingness"]["MinN"]

            # If it's a percentage it will be a string and must be converted.
            if isinstance(amThresh, str):
                amThresh = float(amThresh)

            # Mask based on threshold.
            Mask[n_s_sv<amThresh]=0

    # We remove anything with 1 degree of freedom (or less) by default.
    # 1 degree of freedom seems to cause broadcasting errors on a very
    # small percentage of voxels.
    Mask[n_s_sv<=n_p+1]=0

    if 'analysis_mask' in inputs:

        addmask_path = inputs["analysis_mask"]
        
        # Read in the mask nifti.
        addmask = blmm_load(addmask_path).get_data().reshape([n_v,1])
        
        Mask[addmask==0]=0

    # Output final mask map
    maskmap = nib.Nifti1Image(Mask.reshape(
                                    NIFTIsize[0],
                                    NIFTIsize[1],
                                    NIFTIsize[2]
                                    ),
                              nifti.affine,
                              header=nifti.header)
    nib.save(maskmap, os.path.join(OutDir,'blmm_vox_mask.nii'))
    del maskmap

    # Get indices of voxels in ring around brain where there are
    # missing studies.
    R_inds = np.where((Mask==1)*(n_s_sv<n_s))[0]

    # Get indices of the "inner" volume where all studies had information
    # present. I.e. the voxels (usually near the middle of the brain) where
    # every voxel has a reading for every study.
    I_inds = np.where((Mask==1)*(n_s_sv==n_s))[0]
    del Mask

    # Number of voxels in ring
    n_v_r = R_inds.shape[0]

    # Number of voxels in inner mask
    n_v_i = I_inds.shape[0]

    # Number of voxels in whole (inner + ring) mask
    n_v_m = n_v_i + n_v_r

    # Create dpf map
    df_r = n_s_sv[R_inds,:] - n_p
    df_r = df_r.reshape([n_v_r])
    df_i = n_s - n_p

    # Unmask df
    df = np.zeros([n_v])
    df[R_inds] = df_r
    df[I_inds] = df_i

    df = df.reshape(int(NIFTIsize[0]),
                    int(NIFTIsize[1]),
                    int(NIFTIsize[2]))

    # Save beta map.
    dfmap = nib.Nifti1Image(df,
                            nifti.affine,
                            header=nifti.header)
    nib.save(dfmap, os.path.join(OutDir,'blmm_vox_edf.nii'))
    del df, dfmap

    # ----------------------------------------------------------------------
    # Load X'X, X'Y, Y'Y, X'Z, Y'Z, Z'Z
    # ----------------------------------------------------------------------
    # Read the matrices from the first batch. Note XtY is transposed as np
    # handles lots of rows much faster than lots of columns.
    sumXtY = np.load(os.path.join(OutDir,"tmp","XtY1.npy")).transpose()
    sumYtY = np.load(os.path.join(OutDir,"tmp","YtY1.npy"))
    sumZtY = np.load(os.path.join(OutDir,"tmp","ZtY1.npy"))

    # Work out the uniqueness mask for the spatially varying designs
    uniquenessMask = blmm_load(os.path.join(OutDir,"tmp", 
        "blmm_vox_uniqueM_batch1.nii")).get_data().reshape(n_v)

    # Work out the uniqueness mask inside the ring around the brain
    uniquenessMask_r = uniquenessMask[R_inds]

    # Work out the uniqueness mask value inside the inner part of the brain
    uniquenessMask_i = uniquenessMask[I_inds[0]]

    maxM = np.int32(np.amax(uniquenessMask))

    # read in XtX, ZtX, ZtZ
    ZtZ_batch_unique = np.load(
        os.path.join(OutDir,"tmp","ZtZ1.npy"))
    ZtX_batch_unique = np.load(
        os.path.join(OutDir,"tmp","ZtX1.npy"))
    XtX_batch_unique = np.load(
        os.path.join(OutDir,"tmp","XtX1.npy"))

    # Make zeros for outer ring of brain ZtZ, XtX, ZtX etc
    ZtZ_batch_r = np.zeros((n_v_r, ZtZ_batch_unique.shape[1]))
    ZtX_batch_r = np.zeros((n_v_r, ZtX_batch_unique.shape[1]))
    XtX_batch_r = np.zeros((n_v_r, XtX_batch_unique.shape[1]))

    # Fill with unique maskings
    for m in range(1,maxM+1):

        # Work out Z'Z, Z'X and X'X for the ring
        ZtZ_batch_r[np.where(uniquenessMask_r==m),:] = ZtZ_batch_unique[(m-1),:]
        ZtX_batch_r[np.where(uniquenessMask_r==m),:] = ZtX_batch_unique[(m-1),:]
        XtX_batch_r[np.where(uniquenessMask_r==m),:] = XtX_batch_unique[(m-1),:]

        # Work out Z'Z, Z'X and X'X for the inner
        if uniquenessMask_i == m:
            ZtZ_batch_i = ZtZ_batch_unique[(m-1),:]
            ZtX_batch_i = ZtX_batch_unique[(m-1),:]
            XtX_batch_i = XtX_batch_unique[(m-1),:]

    # Perform summation for ring
    sumXtX_r = XtX_batch_r
    sumZtX_r = ZtX_batch_r
    sumZtZ_r = ZtZ_batch_r

    # Perform summation for ring
    sumXtX_i = XtX_batch_i
    sumZtX_i = ZtX_batch_i
    sumZtZ_i = ZtZ_batch_i

    a = XtX_batch_i[0]

    # Delete the files as they are no longer needed.
    os.remove(os.path.join(OutDir,"tmp","XtY1.npy"))
    os.remove(os.path.join(OutDir,"tmp","YtY1.npy"))
    os.remove(os.path.join(OutDir,"tmp","ZtX1.npy"))
    os.remove(os.path.join(OutDir,"tmp","ZtY1.npy"))
    os.remove(os.path.join(OutDir,"tmp","ZtZ1.npy"))
    os.remove(os.path.join(OutDir,"tmp","blmm_vox_uniqueM_batch1.nii"))

    # Cycle through batches and add together results.
    for batchNo in range(2,(n_b+1)):

        sumXtY = sumXtY + np.load(
            os.path.join(OutDir,"tmp","XtY" + str(batchNo) + ".npy")).transpose()

        sumYtY = sumYtY + np.load(
            os.path.join(OutDir,"tmp","YtY" + str(batchNo) + ".npy"))

        sumZtY = sumZtY + np.load(
            os.path.join(OutDir,"tmp","ZtY" + str(batchNo) + ".npy"))
        
        # Read in uniqueness Mask file
        uniquenessMask = blmm_load(os.path.join(OutDir,"tmp", 
            "blmm_vox_uniqueM_batch" + str(batchNo) + ".nii")).get_data().reshape(n_v)

        # Work out the uniqueness mask inside the ring around the brain
        uniquenessMask_r = uniquenessMask[R_inds]

        # Work out the uniqueness mask value inside the inner part of the brain
        uniquenessMask_i = uniquenessMask[I_inds[0]]


        maxM = np.int32(np.amax(uniquenessMask))

        # read in XtX, ZtX, ZtZ
        ZtZ_batch_unique = np.load(
            os.path.join(OutDir,"tmp","ZtZ" + str(batchNo) + ".npy"))
        ZtX_batch_unique = np.load(
            os.path.join(OutDir,"tmp","ZtX" + str(batchNo) + ".npy"))
        XtX_batch_unique = np.load(
            os.path.join(OutDir,"tmp","XtX" + str(batchNo) + ".npy"))

        # Make zeros for whole nifti ZtZ, XtX, ZtX etc
        ZtZ_batch_r = np.zeros((n_v_r, ZtZ_batch_unique.shape[1]))
        ZtX_batch_r = np.zeros((n_v_r, ZtX_batch_unique.shape[1]))
        XtX_batch_r = np.zeros((n_v_r, XtX_batch_unique.shape[1]))

        # Fill with unique maskings
        for m in range(1,maxM+1):

            ZtZ_batch_r[np.where(uniquenessMask_r==m),:] = ZtZ_batch_unique[(m-1),:]
            ZtX_batch_r[np.where(uniquenessMask_r==m),:] = ZtX_batch_unique[(m-1),:]
            XtX_batch_r[np.where(uniquenessMask_r==m),:] = XtX_batch_unique[(m-1),:]

            # Work out Z'Z, Z'X and X'X for the inner
            if uniquenessMask_i == m:

                ZtZ_batch_i = ZtZ_batch_unique[(m-1),:]
                ZtX_batch_i = ZtX_batch_unique[(m-1),:]
                XtX_batch_i = XtX_batch_unique[(m-1),:]

        # Add to running total
        sumXtX_r = sumXtX_r + XtX_batch_r
        sumZtX_r = sumZtX_r + ZtX_batch_r
        sumZtZ_r = sumZtZ_r + ZtZ_batch_r

        sumXtX_i = sumXtX_i + XtX_batch_i
        sumZtX_i = sumZtX_i + ZtX_batch_i
        sumZtZ_i = sumZtZ_i + ZtZ_batch_i

        a = a + XtX_batch_i[0]
        
        # Delete the files as they are no longer needed.
        os.remove(os.path.join(OutDir, "tmp","XtY" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp","YtY" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp","ZtY" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp","XtX" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp","ZtX" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp","ZtZ" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp", "blmm_vox_uniqueM_batch" + str(batchNo) + ".nii"))

    # ----------------------------------------------------------------------
    # Calculate betahat = (X'X)^(-1)X'Y and output beta maps
    # ----------------------------------------------------------------------    

    # Reshaping
    sumXtY = sumXtY.transpose()

    sumXtY = sumXtY.reshape([n_v, n_p, 1])
    sumYtY = sumYtY.reshape([n_v, 1, 1])
    sumZtY = sumZtY.reshape([n_v, n_q, 1])

    sumXtX_r = sumXtX_r.reshape([n_v_r, n_p, n_p])
    sumZtX_r = sumZtX_r.reshape([n_v_r, n_q, n_p])
    sumZtZ_r = sumZtZ_r.reshape([n_v_r, n_q, n_q])

    sumXtX_i = sumXtX_i.reshape([1, n_p, n_p])
    sumZtX_i = sumZtX_i.reshape([1, n_q, n_p])
    sumZtZ_i = sumZtZ_i.reshape([1, n_q, n_q])

    # Empty vectors for parameter estimates
    beta = np.zeros([n_v, n_p])
    sigma2 = np.zeros([n_v, 1]) 

    REML = False

    # If we have indices where only some studies are present, work out X'X and
    # X'Y for these studies.
    if n_v_r:

        # Calculate masked X'Y for ring
        XtY_r = sumXtY[R_inds,:,:]

        # Calculate Y'Y for ring
        YtY_r = sumYtY[R_inds,:,:]

        # Calculate masked Z'Y for ring
        ZtY_r = sumZtY[R_inds,:,:]

        # We rename these for convinience
        XtX_r = sumXtX_r
        ZtZ_r = sumZtZ_r
        ZtX_r = sumZtX_r

        # We calculate these by transposing
        YtX_r = XtY_r.transpose((0,2,1))
        YtZ_r = ZtY_r.transpose((0,2,1))
        XtZ_r = ZtX_r.transpose((0,2,1))

        # Spatially varying nv for ring
        n_s_sv_r = n_s_sv[R_inds,:]

        # Clear some memory
        del sumXtX_r, sumZtX_r, sumZtZ_r

        #================================================================================
        # Run parameter estimation
        #================================================================================
        paramVec_r = pSFS(XtX_r, XtY_r, ZtX_r, ZtY_r, ZtZ_r, XtZ_r, YtZ_r, YtY_r, YtX_r, nlevels, nparams, 1e-6,n_s_sv_r,reml=REML)



    # If we have indices where all studies are present, work out X'X and
    # X'Y for these studies.
    if n_v_i:
        
        # X'X must be 1 by np by np for broadcasting
        XtX_i = sumXtX_i.reshape([1, n_p, n_p])

        XtY_i = sumXtY[I_inds,:]

        # Calculate Y'Y for inner
        YtY_i = sumXtY[I_inds,:]

        # Calculate Y'Y for inner
        YtY_i = sumYtY[I_inds,:,:]

        # Calculate masked Z'X for inner
        ZtX_i = sumZtX_i.reshape([1, n_q, n_p])

        # Calculate masked Z'Y for inner
        ZtY_i = sumZtY[I_inds,:,:]

        # Calculate Z'Y for inner
        ZtZ_i = sumZtZ_i.reshape([1, n_q, n_q])

        # We calculate these by transposing
        YtX_i = XtY_i.transpose((0,2,1))
        YtZ_i = ZtY_i.transpose((0,2,1))
        XtZ_i = ZtX_i.transpose((0,2,1))

        # Clear some memory
        del sumXtX_i, sumZtX_i, sumZtZ_i
        del sumXtY, sumYtY, sumZtY

        #================================================================================
        # Run parameter estimation
        #================================================================================
        paramVec_i = pSFS(XtX_i, XtY_i, ZtX_i, ZtY_i, ZtZ_i, XtZ_i, YtZ_i, YtY_i, YtX_i, nlevels, nparams, 1e-6,n_s, reml=REML)


    paramVec = np.zeros([n_v, n_p + 1 + np.sum(nparams*(nparams+1)//2)])

    # Complete parameter vector
    if n_v_r:
        paramVec[R_inds,:] = paramVec_r[:].reshape(paramVec[R_inds,:].shape)
        # Assign betas
        beta_r = paramVec_r[:, 0:n_p]
        beta[R_inds,:] = beta_r.reshape([n_v_r, n_p])

    if n_v_i:

        paramVec[I_inds,:] = paramVec_i[:].reshape(paramVec[I_inds,:].shape)

        beta_i = paramVec_i[:, 0:n_p]
        beta[I_inds,:] = beta_i.reshape([n_v_i, n_p])

    beta = beta.reshape([n_v, n_p]).transpose()

    beta_out = np.zeros([int(NIFTIsize[0]),
                         int(NIFTIsize[1]),
                         int(NIFTIsize[2]),
                         beta.shape[0]])

    # Cycle through betas and output results.
    for k in range(0,beta.shape[0]):

        beta_out[:,:,:,k] = beta[k,:].reshape(int(NIFTIsize[0]),
                                              int(NIFTIsize[1]),
                                              int(NIFTIsize[2]))

    # Save beta map.
    betamap = nib.Nifti1Image(beta_out,
                              nifti.affine,
                              header=nifti.header)
    nib.save(betamap, os.path.join(OutDir,'blmm_vox_beta.nii'))
    del beta_out, betamap

    if np.ndim(beta) == 0:
        beta = np.array([[beta]])
    elif np.ndim(beta) == 1:
        beta = np.array([beta])

    # Get the D matrices
    FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)//2) + n_p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,n_p+1)


    if n_v_r:

        sigma2_r = paramVec_r[:,n_p:(n_p+1),:]

        Ddict_r = dict()
        # D as a dictionary
        for k in np.arange(len(nparams)):

            Ddict_r[k] = vech2mat3D(paramVec_r[:,FishIndsDk[k]:FishIndsDk[k+1],:])
          
        # Full version of D
        D_r = getDfromDict3D(Ddict_r, nparams, nlevels)

        # ----------------------------------------------------------------------
        # Calculate log-likelihood
        # ---------------------------------------------------------------------- 

        # Variables for likelihood
        DinvIplusZtZD_r = D_r @ np.linalg.inv(np.eye(n_q) + ZtZ_r @ D_r)
        Zte_r = ZtY_r - (ZtX_r @ beta_r)
        ete_r = ssr3D(YtX_r, YtY_r, XtX_r, beta_r)

        # Output log likelihood
        llh_r = llh3D(n_s_sv_r, ZtZ_r, Zte_r, ete_r, sigma2_r, DinvIplusZtZD_r, D_r, REML, XtX_r, XtZ_r, ZtX_r) - (0.5*(n_s_sv_r)*np.log(2*np.pi)).reshape(ete_r.shape[0])


    if n_v_i:

        sigma2_i = paramVec_i[:,n_p:(n_p+1),:]

        Ddict_i = dict()
        # D as a dictionary
        for k in np.arange(len(nparams)):

            Ddict_i[k] = makeDnnd3D(vech2mat3D(paramVec_i[:,FishIndsDk[k]:FishIndsDk[k+1],:]))
          
        # Full version of D
        D_i = getDfromDict3D(Ddict_i, nparams, nlevels)

        # ----------------------------------------------------------------------
        # Calculate log-likelihood
        # ---------------------------------------------------------------------- 

        # Variables for likelihood
        DinvIplusZtZD_i = D_i @ np.linalg.inv(np.eye(n_q) + ZtZ_i @ D_i)
        Zte_i = ZtY_i - (ZtX_i @ beta_i)
        ete_i = ssr3D(YtX_i, YtY_i, XtX_i, beta_i)

        # Output log likelihood
        llh_i = llh3D(n_s, ZtZ_i, Zte_i, ete_i, sigma2_i, DinvIplusZtZD_i, D_i, REML, XtX_i, XtZ_i, ZtX_i) - 0.5*(n_s)*np.log(2*np.pi)

    # Unmask llh
    llh = np.zeros([n_v,1])
    if n_v_r:

        llh[R_inds,:] = llh_r[:].reshape(llh[R_inds,:].shape)

    if n_v_i:

        llh[I_inds,:] = llh_i[:].reshape(llh[I_inds,:].shape)
    


    llh_out = llh.reshape(int(NIFTIsize[0]),
                          int(NIFTIsize[1]),
                          int(NIFTIsize[2]))

    # Save beta map.
    llhmap = nib.Nifti1Image(llh_out,
                             nifti.affine,
                             header=nifti.header)
    nib.save(llhmap, os.path.join(OutDir,'blmm_vox_llh.nii'))
    del llhmap, llh_out, llh_i, llh_r


    # Unmask sigma2
    if n_v_r:

        sigma2[R_inds,:] = sigma2_r[:].reshape(sigma2[R_inds,:].shape)

    if n_v_i:

        sigma2[I_inds,:] = sigma2_i[:].reshape(sigma2[I_inds,:].shape)



    sigma2_out = sigma2.reshape(int(NIFTIsize[0]),
                                int(NIFTIsize[1]),
                                int(NIFTIsize[2]))

    # Save beta map.
    sigma2map = nib.Nifti1Image(sigma2_out,
                                nifti.affine,
                                header=nifti.header)
    nib.save(sigma2map, os.path.join(OutDir,'blmm_vox_sigma2.nii'))

    # Save D
    vechD = np.zeros([n_v, n_q_u])

    if n_v_r:

        vechD[R_inds,:] = paramVec_r[:,(n_p+1):,:].reshape(vechD[R_inds,:].shape)

    if n_v_i:

        vechD[I_inds,:] = paramVec_i[:,(n_p+1):,:].reshape(vechD[I_inds,:].shape)

    # Output vechD
    vechD = vechD.reshape([n_v, n_q_u]).transpose()

    vechD_out = np.zeros([int(NIFTIsize[0]),
                         int(NIFTIsize[1]),
                         int(NIFTIsize[2]),
                         vechD.shape[0]])

    # Cycle through betas and output results.
    for k in range(0,vechD.shape[0]):

        vechD_out[:,:,:,k] = vechD[k,:].reshape(int(NIFTIsize[0]),
                                               int(NIFTIsize[1]),
                                               int(NIFTIsize[2]))

    # Save beta map.
    vechDmap = nib.Nifti1Image(vechD_out,
                               nifti.affine,
                               header=nifti.header)
    nib.save(vechDmap, os.path.join(OutDir,'blmm_vox_D.nii'))

    # ----------------------------------------------------------------------
    # Calculate residual mean squares = e'e/(n_s - n_p)
    #
    # Note: In the mixed model resms is different to our sigma2 estimate as:
    #
    #  - resms = e'e/(n-p)
    #  - sigma2 = e'V^(-1)e/n for "Simplified methods" or has no closed form
    #             expression for more general methods
    #
    # ----------------------------------------------------------------------

    if n_v_r:

        # In spatially varying the degrees of freedom
        # varies across voxels
        resms_r = get_resms3D(YtX_r, YtY_r, XtX_r, beta_r,n_s_sv_r-n_p).reshape(n_v_r)
        addBlockToNifti(os.path.join(OutDir, 'blmm_vox_resms.nii'), resms_r, R_inds,volc=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)


    if n_v_i:

        # All voxels in the inner mask have n_s scans present
        resms_i = get_resms3D(YtX_i, YtY_i, XtX_i, beta_i, n_s-n_p).reshape(n_v_i)
        addBlockToNifti(os.path.join(OutDir, 'blmm_vox_resms.nii'), resms_i, I_inds,volc=0,dim=NIFTIsize,aff=nifti.affine,hdr=nifti.header)


    # ----------------------------------------------------------------------
    # Calculate beta covariance maps
    # ----------------------------------------------------------------------

    if "OutputCovB" in inputs:
        OutputCovB = inputs["OutputCovB"]
    else:
        OutputCovB = True

    if OutputCovB:

        dimCov =  (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],n_p**2)
        
        # Output variance for each pair of betas
        if n_v_r: 

            # Work out cov beta
            covB_r = get_covB3D(XtX_r, XtZ_r, DinvIplusZtZD_r, sigma2_r).reshape(n_v_r, n_p**2)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_cov.nii'), covB_r, R_inds,volc=None,dim=dimCov,aff=nifti.affine,hdr=nifti.header)
            del covB_r

        if n_v_i: 

            # Work out cov beta
            covB_i = get_covB3D(XtX_i, XtZ_i, DinvIplusZtZD_i, sigma2_i).reshape(n_v_i, n_p**2)
            addBlockToNifti(os.path.join(OutDir, 'blmm_vox_cov.nii'), covB_i, I_inds,volc=None,dim=dimCov,aff=nifti.affine,hdr=nifti.header)
            del covB_i

    # ----------------------------------------------------------------------
    # Calculate COPEs, statistic maps and covariance maps.
    # ----------------------------------------------------------------------
    n_c = len(inputs['contrasts'])

    # Record how many T contrasts and F contrasts we have seen
    n_ct = 0
    n_cf = 0
    for i in range(0,n_c):

        # Read in contrast vector
        L = blmm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        L = np.array(L)

        if L.ndim == 1:
            n_ct = n_ct + 1
        else:
            n_cf = n_cf + 1

    # Current number for contrast (T and F)
    current_n_ct = 0
    current_n_cf = 0


    for i in range(0,n_c):

        # Read in contrast vector
        # Get number of parameters
        L = blmm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        L = np.array(L)
    
        if L.ndim == 1:
            statType='T'
            L = L.reshape([1,L.shape[0]])
        else:
            statType='F'

        if statType == 'T':

            # Work out the dimension of the T-stat-related volumes
            dimT = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],n_ct)


            # Calculate masked T statistic image for ring
            if n_v_r:

                # Work out L\beta
                Lbeta_r = L @ beta_r
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_con.nii'), Lbeta_r, R_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)
    
                # Work out s.e.(L\beta)
                seLB_r = np.sqrt(get_varLB3D(L, XtX_r, XtZ_r, DinvIplusZtZD_r, sigma2_r).reshape(n_v_r))
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conSE.nii'), seLB_r, R_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)
    

                # Calculate sattherwaite estimate of the degrees of freedom of this statistic
                swdfc_r = get_swdf_T3D(L, D_r, sigma2_r, ZtX_r, ZtY_r, XtX_r, ZtZ_r, XtY_r, YtX_r, YtZ_r, XtZ_r, YtY_r, n_s_sv_r, nlevels, nparams).reshape(n_v_r)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conT_swedf.nii'), swdfc_r, R_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                # Obtain and output T statistic
                Tc_r = get_T3D(L, XtX_r, XtZ_r, DinvIplusZtZD_r, beta_r, sigma2_r).reshape(n_v_r)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conT.nii'), Tc_r, R_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                # Obatin and output p-values
                pc_r = T2P3D(Tc_r,swdfc_r,inputs)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conTlp.nii'), pc_r, R_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)


            if n_v_i:

                # Work out L\beta
                Lbeta_i = L @ beta_i
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_con.nii'), Lbeta_i, I_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                # Work out s.e.(L\beta)
                seLB_i = np.sqrt(get_varLB3D(L, XtX_i, XtZ_i, DinvIplusZtZD_i, sigma2_i).reshape(n_v_i))
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conSE.nii'), seLB_i, I_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                # Calculate sattherwaite estimate of the degrees of freedom of this statistic
                swdfc_i = get_swdf_T3D(L, D_i, sigma2_i, ZtX_i, ZtY_i, XtX_i, ZtZ_i, XtY_i, YtX_i, YtZ_i, XtZ_i, YtY_i, n_s, nlevels, nparams).reshape(n_v_i)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conT_swedf.nii'), swdfc_i, I_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)

                # Obtain and output T statistic
                Tc_i = get_T3D(L, XtX_i, XtZ_i, DinvIplusZtZD_i, beta_i, sigma2_i).reshape(n_v_i)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conT.nii'), Tc_i, I_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)
                
                # Obtain and output p-values
                pc_i = T2P3D(Tc_i,swdfc_i,inputs)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conTlp.nii'), pc_i, I_inds,volc=current_n_ct,dim=dimT,aff=nifti.affine,hdr=nifti.header)


            # Record that we have seen another T contrast
            current_n_ct = current_n_ct + 1

            print('whole of T ran')


        if statType == 'F':

            # Work out the dimension of the F-stat-related volumes
            dimF = (NIFTIsize[0],NIFTIsize[1],NIFTIsize[2],n_cf)

            # Calculate the numerator of the F statistic for the ring
            if n_v_r:

                # Calculate sattherthwaite degrees of freedom for the inner.
                swdfc_r = get_swdf_F3D(L, D_r, sigma2_r, ZtX_r, ZtY_r, XtX_r, ZtZ_r, XtY_r, YtX_r, YtZ_r, XtZ_r, YtY_r, n_s_sv_r, nlevels, nparams).reshape(n_v_r)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conF_swedf.nii'), swdfc_r, R_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                # Calculate F statistic.
                Fc_r=get_F3D(L, XtX_r, XtZ_r, DinvIplusZtZD_r, beta_r, sigma2_r).reshape(n_v_r)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conF.nii'), Fc_r, R_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                # Work out p for this contrast
                pc_r = F2P3D(Fc_r, L, swdfc_r, inputs).reshape(n_v_r)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conFlp.nii'), pc_r, R_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)
 
                # Calculate partial R2 masked for ring.
                R2_r = get_R23D(L, Fc_r, swdfc_r).reshape(n_v_r)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conR2.nii'), R2_r, R_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)


            # Calculate the numerator of the F statistic for the inner 
            if n_v_i:

                # Calculate sattherthwaite degrees of freedom for the inner.
                swdfc_i = get_swdf_F3D(L, D_i, sigma2_i, ZtX_i, ZtY_i, XtX_i, ZtZ_i, XtY_i, YtX_i, YtZ_i, XtZ_i, YtY_i, n_s, nlevels, nparams).reshape(n_v_i)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conF_swedf.nii'), swdfc_i, I_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)
                
                # Calculate F statistic.
                Fc_i=get_F3D(L, XtX_i, XtZ_i, DinvIplusZtZD_i, beta_i, sigma2_i).reshape(n_v_i)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conF.nii'), Fc_i, I_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                # Work out p for this contrast
                pc_i = F2P3D(Fc_i, L, swdfc_i, inputs).reshape(n_v_i)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conFlp.nii'), pc_i, I_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

                # Calculate partial R2 masked for inner mask.
                R2_i = get_R23D(L, Fc_i, swdfc_i).reshape(n_v_i)
                addBlockToNifti(os.path.join(OutDir, 'blmm_vox_conR2.nii'), R2_i, I_inds,volc=current_n_cf,dim=dimF,aff=nifti.affine,hdr=nifti.header)

            # Record that we have seen another F contrast
            current_n_cf = current_n_cf + 1

    # Clean up files
    if len(args)==0:
        os.remove(os.path.join(OutDir, 'nb.txt'))
    shutil.rmtree(os.path.join(OutDir, 'tmp'))

    w.resetwarnings()


def addBlockToNifti(fname, block, blockInds,dim=None,volc=None,aff=None,hdr=None):

    # Check volc is correct datatype
    if volc is not None:

        volc = int(volc)

    # # Work out dimension of block
    # if np.ndim(block)>1:
    #     if block.shape[1]==1:
    #         block = block.reshape(block.shape[0])

    # Check whether the NIFTI exists already
    if os.path.isfile(fname):

        # Load in NIFTI
        img = nib.load(fname)

        # Work out dim if we don't already have it
        dim = img.shape

        # Work out data
        data = img.get_fdata()

        # Work out affine
        affine = img.affine
        
    else:

        # If we know how, make the NIFTI
        if dim is not None:
            
            # Make data
            data = np.zeros(dim)

            # Make affine
            if aff is None:
                affine = np.eye(4)
            else:
                affine = aff

        else:

            # Throw an error because we don't know what to do
            raise Exception('NIFTI does not exist and dimensions not given')

    # Work out the number of output volumes inside the nifti 
    if len(dim)==3:

        # We only have one volume in this case
        n_vol = 1
        dim = np.array([dim[0],dim[1],dim[2],1])

    else:

        # The number of volumes is the last dimension
        n_vol = dim[3]

    # Seperate copy of data for outputting
    data_out = np.array(data).reshape(dim)

    # Work out the number of voxels
    n_vox = np.prod(dim[:3])

    # Reshape     
    data = data.reshape([n_vox, n_vol])

    # Add all the volumes
    if volc is None:

        # Add block
        data[blockInds,:] = block.reshape(data[blockInds,:].shape)
        
        # Cycle through volumes, reshaping.
        for k in range(0,data.shape[1]):

            data_out[:,:,:,k] = data[:,k].reshape(int(dim[0]),
                                                  int(dim[1]),
                                                  int(dim[2]))

    # Add the one volume in the correct place
    else:

        # We're only looking at this volume
        data = data[:,volc].reshape((n_vox,1))

        # Add block
        data[blockInds,:] = block.reshape(data[blockInds,:].shape)
        
        # Put in the volume
        data_out[:,:,:,volc] = data[:,0].reshape(int(dim[0]),
                                                 int(dim[1]),
                                                 int(dim[2]))


    # Make NIFTI
    nifti = nib.Nifti1Image(data_out, affine, header=hdr)
    
    # Save NIFTI
    nib.save(nifti, fname)

    del nifti, fname, data_out, affine



# This function takes in two matrices of dimension
# n_v_i by k and n_v_r by k and two sets of indices.
def outputNifti(vol_i,vol_r,I_inds,R_inds,dimv,fpath):

    # Work out n_v_r and n_v_i
    n_v_r = R_inds.shape[0]
    n_v_i = I_inds.shape[0]

    # Work out number of volumes to be output
    if ndim(vol_i)==2:
        n_o = vol_i.shape[1]
    else:
        n_o = 1

    # Number of voxels
    n_v = np.prod(dimv)

    # Initiate empty nifti
    vol = np.zeros([n_v,n_o])

    # Put vol_r and vol_i into the volume
    if n_v_r:

        vol[R_inds,:] = vol_r.reshape(vol[R_inds,:].shape)

    if n_v_i:

        vol[I_inds,:] = vol_i.reshape(vol[I_inds,:].shape)

    
    # Output volume
    vol = vol.transpose()
    vol_out = np.zeros([int(dimv[0]),
                        int(dimv[1]),
                        int(dimv[2]),
                        vol.shape[0]])

    # Cycle through individual niftis and output results.
    for k in range(0,vechD.shape[0]):

        vol_out[:,:,:,k] = vol[k,:].reshape(int(dimv[0]),
                                            int(dimv[1]),
                                            int(dimv[2]))

    # Save final volume.
    finalVol = nib.Nifti1Image(vol_out,
                               nifti.affine,
                               header=nifti.header)
    nib.save(finalVol, fpath)

def get_resms3D(YtX, YtY, XtX, beta,n):

    ete = ssr3D(YtX, YtY, XtX, beta)

    # Reshape n if necessary
    if isinstance(n,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(n.shape)>1:
    
            n = n.reshape(ete.shape)

    return(ete/n)

def get_covB3D(XtX, XtZ, DinvIplusZtZD, sigma2):

    # Reshape n if necessary
    if isinstance(sigma2,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(sigma2.shape)>1:
    
            sigma2 = sigma2.reshape(sigma2.shape[0])

    # Work out X'V^{-1}X = X'X - X'ZD(I+Z'ZD)^{-1}Z'X
    XtinvVX = XtX - XtZ @ DinvIplusZtZD @ XtZ.transpose((0,2,1))

    # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
    covB = np.linalg.inv(XtinvVX)

    # Calculate sigma^2(X'V^{-1}X)^(-1)
    covB = np.einsum('i,ijk->ijk',sigma2,covB)

    # Return result
    return(covB)

def get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2):

    # Reshape n if necessary
    if isinstance(sigma2,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(sigma2.shape)>1:
    
            sigma2 = sigma2.reshape(sigma2.shape[0])

    # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
    varLB = L @ get_covB3D(XtX, XtZ, DinvIplusZtZD, sigma2) @ L.transpose()

    # Return result
    return(varLB)

def get_R23D(L, F, df):

    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Convert F to R2
    R2 = (rL*F)/(rL*F + df)
    
    # Return R2
    return(R2)


def get_T3D(L, XtX, XtZ, DinvIplusZtZD, betahat, sigma2):

    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Work out Lbeta
    LB = L @ betahat

    # Work out se(T)
    varLB = get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2)

    # Work out T
    T = LB/np.sqrt(varLB)

    # Return T
    return(T)


def get_F3D(L, XtX, XtZ, DinvIplusZtZD, betahat, sigma2):

    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Work out Lbeta
    LB = L @ betahat

    # Work out se(F)
    varLB = get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2)

    # Work out F
    F = LB.transpose(0,2,1) @ np.linalg.inv(varLB) @ LB/rL

    # Return T
    return(F)
    

def T2P3D(T,df,inputs):

    # Initialize empty P
    P = np.zeros(np.shape(T))

    # Do this seperately for >0 and <0 to avoid underflow
    P[T < 0] = -np.log10(1-stats.t.cdf(T[T < 0], df[T < 0]))
    P[T >= 0] = -np.log10(stats.t.cdf(-T[T >= 0], df[T >= 0]))

    # Remove infs
    if "minlog" in inputs:
        P[np.logical_and(np.isinf(P), P<0)]=inputs['minlog']
    else:
        P[np.logical_and(np.isinf(P), P<0)]=-323.3062153431158

    return(P)


def F2P3D(F, L, df_denom, inputs):
    
    # Get the rank of L
    df_num = np.linalg.matrix_rank(L)

    # Work out P
    P = -np.log10(1-stats.f.cdf(F, df_num, df_denom))

    # Remove infs
    if "minlog" in inputs:
        P[np.logical_and(np.isinf(P), P<0)]=inputs['minlog']
    else:
        P[np.logical_and(np.isinf(P), P<0)]=-323.3062153431158

    return(P)


# ============================================================================================================
#
# WIP: SATTHERTHWAITE INTEGRATING
#
# ============================================================================================================

def get_swdf_F3D(L, D, sigma2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, nlevels, nparams): 

    # Reshape sigma2 if necessary
    sigma2 = sigma2.reshape(sigma2.shape[0])

    # Reshape n if necessary
    if isinstance(n,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(n.shape)>1:
    
            n = n.reshape(sigma2.shape)

    # L is rL in rank
    rL = np.linalg.matrix_rank(L)

    # Initialize empty sum.
    sum_swdf_adj = np.zeros(sigma2.shape)

    # Loop through first rL rows of L
    for i in np.arange(rL):

        # Work out the swdf for each row of L
        swdf_row = get_swdf_T3D(L[i:(i+1),:], D, sigma2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, nlevels, nparams)

        # Work out adjusted df = df/(df-2)
        swdf_adj = swdf_row/(swdf_row-2)

        # Add to running sum
        sum_swdf_adj = sum_swdf_adj + swdf_adj.reshape(sum_swdf_adj.shape)

    # Work out final df
    df = 2*sum_swdf_adj/(sum_swdf_adj-rL)

    # Return df
    return(df)

def get_swdf_T3D(L, D, sigma2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, nlevels, nparams): 

    # Reshape sigma2 if necessary
    sigma2 = sigma2.reshape(sigma2.shape[0])

    # Reshape n if necessary
    if isinstance(n,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(n.shape)>1:
    
            n = n.reshape(sigma2.shape)

    # Get D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(ZtZ.shape[1]) + ZtZ @ D)

    # Get S^2
    S2 = get_S23D(L, XtX, XtZ, DinvIplusZtZD, sigma2)
    
    # Get derivative of S^2
    dS2 = get_dS23D(nparams, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2)

    # Get Fisher information matrix
    InfoMat = get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nparams, ZtZ)

    # Calculate df estimator
    df = 2*(S2**2)/(dS2.transpose(0,2,1) @ np.linalg.inv(InfoMat) @ dS2)

    # Return df
    return(df)



def get_S23D(L, XtX, XtZ, DinvIplusZtZD, sigma2):

    # Calculate sigma^2L(X'V^{-1}X)^(-1)L'
    varLB = get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2)

    return(varLB)


def get_dS23D(nparams, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2):

    # ZtX
    ZtX = XtZ.transpose(0,2,1)

    # Number of voxels
    nv = DinvIplusZtZD.shape[0]

    # Calculate X'V^{-1}X=X'(I+ZDZ')^{-1}X=X'X-X'Z(I+DZ'Z)^{-1}DZ'X
    XtiVX = XtX - XtZ @  DinvIplusZtZD @ ZtX

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2 = np.zeros((nv, 1+np.int32(np.sum(nparams*(nparams+1)/2)),1))

    # Work out indices for each start of each component of vector 
    # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
    DerivInds = np.int32(np.cumsum(nparams*(nparams+1)/2) + 1)
    DerivInds = np.insert(DerivInds,0,1)

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.inv(XtiVX) @ L.transpose()

    # Add to dS2
    dS2[:,0:1] = dS2dsigma2.reshape(dS2[:,0:1].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nparams)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((np.int32(nparams[k]*(nparams[k]+1)/2),1))#...

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nparams)
                    
            # Work out Z_(k,j)'Z
            ZkjtZ = ZtZ[:,Ikj,:]

            # Work out Z_(k,j)'X
            ZkjtX = ZtX[:,Ikj,:]

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = ZkjtX - ZkjtZ @ DinvIplusZtZD @ ZtX

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.inv(XtiVX) @ L.transpose()
            
            # Sum terms
            dS2dvechDk = dS2dvechDk + mat2vech3D(kron3D(K,K.transpose(0,2,1)))

        # Multiply by sigma^2
        dS2dvechDk = np.einsum('i,ijk->ijk',sigma2,dS2dvechDk)

        # Add to dS2
        dS2[:,DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2[:,DerivInds[k]:DerivInds[k+1]].shape)

    return(dS2)

def get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nparams, ZtZ):

    # Number of random effects, q
    q = np.sum(np.dot(nparams,nlevels))

    # Number of voxels 
    nv = sigma2.shape[0]

    # Duplication matrices
    # ------------------------------------------------------------------------------
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):

        invDupMatdict[i] = np.asarray(invDupMat2D(nparams[i]).todense())

    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(1 + np.sum(nparams*(nparams+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + 1)
    FishIndsDk = np.insert(FishIndsDk,0,1)

    # Initialize FIsher Information matrix
    FisherInfoMat = np.zeros((nv,tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2**2))
    
    # Add dl/dsigma2 covariance
    FisherInfoMat[:,0,0] = covdldsigma2

    
    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nparams)):

        # Get covariance of dldsigma and dldD      
        covdldsigmadD = get_covdldDkdsigma23D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict).reshape(nv,FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FisherInfoMat[:,0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
        FisherInfoMat[:,FishIndsDk[k]:FishIndsDk[k+1],0:1] = FisherInfoMat[:,0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose((0,2,1))
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nparams)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk23D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict)

            # Add to FImat
            FisherInfoMat[np.ix_(np.arange(nv), IndsDk1, IndsDk2)] = covdldDk1dDk2
            FisherInfoMat[np.ix_(np.arange(nv), IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(np.arange(nv), IndsDk1, IndsDk2)].transpose((0,2,1))


    # Return result
    return(FisherInfoMat)

if __name__ == "__main__":
    main()
