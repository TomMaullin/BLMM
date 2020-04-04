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
import lib.blmm_inference as blmm_inference
import lib.blmm_estimate as blmm_estimate

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

    # Create df map
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


    # Complete parameter vector
    if n_v_r:

        # Run parameter estimation
        beta_r, sigma2_r, D_r = blmm_estimate.main(inputs, R_inds, XtX_r, XtY_r, ZtX_r, ZtY_r, ZtZ_r, XtZ_r, YtZ_r, YtY_r, YtX_r, n_s_sv_r, nlevels, nparams)

        # Run inference
        blmm_inference.main(inputs, nparams, nlevels, R_inds, beta_r, D_r, sigma2_r, n_s_sv_r, XtX_r, XtY_r, XtZ_r, YtX_r, YtY_r, YtZ_r, ZtX_r, ZtY_r, ZtZ_r)       
        
    if n_v_i:

        # Run parameter estimation
        beta_i, sigma2_i, D_i = blmm_estimate.main(inputs, I_inds, XtX_i, XtY_i, ZtX_i, ZtY_i, ZtZ_i, XtZ_i, YtZ_i, YtY_i, YtX_i, n_s, nlevels, nparams)

        # Run inference
        blmm_inference.main(inputs, nparams, nlevels, I_inds, beta_i, D_i, sigma2_i, n_s, XtX_i, XtY_i, XtZ_i, YtX_i, YtY_i, YtZ_i, ZtX_i, ZtY_i, ZtZ_i)

    # Clean up files
    if len(args)==0:
        os.remove(os.path.join(OutDir, 'nb.txt'))
    shutil.rmtree(os.path.join(OutDir, 'tmp'))

    w.resetwarnings()

if __name__ == "__main__":
    main()
