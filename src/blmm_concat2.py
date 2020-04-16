import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
import time
np.set_printoptions(threshold=np.nan)
from lib.npMatrix3d import *
from lib.npMatrix2d import *
from lib.fileio import *
import src.blmm_inference as blmm_inference
import src.blmm_estimate as blmm_estimate

# ====================================================================================
#
# This file is the third stage of the BLMM pipeline. This stage reads in the product
# matrices output by each of the `blmm_batch` jobs during the second stage and s 
# them to obtain the product matrices for the overall model. It also calculates n_sv
# for the whole model and the overall mask.
#
# Following this, the `blmm_concat` code then seperates the voxels in the brain into
# two categories; "inner" and "ring" (explained in the developer notes below). Once
# this has been done the product matrices corresponding to "inner" and "ring" voxels
# are passed to `blmm_estimate`, which estimates the parameters of the model; beta,
# sigma2 and D. Following this, the product matrices and parameter estimates are 
# passed to `blmm_inference`, which generates statistic maps and other miscelanoues 
# output.
#
# ------------------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited: 04/04/2020)
#
# ------------------------------------------------------------------------------------
#
# The code takes the following inputs:
#
#  - ipath (optional): If specified, the first argument will be ased to be a
#                           path to an `inputs` yml file, following the same 
#                           formatting guidelines as `blmm_config.yml`. If not 
#                           specified, the default file `blmm_config.yml` will be 
#                           ased to contain the inputs.
#
# MARKER TODO
#
# ------------------------------------------------------------------------------------
# Developer notes:
# ------------------------------------------------------------------------------------
# In the following code I have used the following subscripts to indicate:
#
# _r - This means this is an array of values corresponding to voxels which
#      are present in between k and n-1 studies (inclusive), where k is
#      decided by the user specified thresholds. These voxels will typically
#      be on the edge of the brain and look like a "ring" around the brain,
#      hence "_r" for ring.
# 
# _i - This means that this is an array of values corresponding to voxels 
#      which are present in all n studies. These will usually look like
#      a smaller mask place inside the whole study mask. Hence "_i" for 
#      inner.
#
# _sv - This means this variable is spatially varying (There is a reading
#       per voxel). 
#
# ====================================================================================
def main(ipath, vb):

    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Inputs file is first argument
    with open(os.path.join(ipath), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # Voxel batch
    vb = int(vb)

    # Check if the maximum memory is saved.    
    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    # --------------------------------------------------------------------------------
    # Read basic inputs
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']

    # Random factor variables.
    rfxmats = inputs['Z']

    # Number of random effects
    r = len(rfxmats)

    # Number of random effects for each factor, q
    nraneffs = []

    # Number of levels for each factor, l
    nlevels = []

    for k in range(r):

        rfxdes = loadFile(rfxmats[k]['f' + str(k+1)]['design'])
        rfxfac = loadFile(rfxmats[k]['f' + str(k+1)]['factor'])

        nraneffs = nraneffs + [rfxdes.shape[1]]
        nlevels = nlevels + [len(np.unique(rfxfac))]

    # Get number of random effects
    nraneffs = np.array(nraneffs)
    nlevels = np.array(nlevels)
    q = np.sum(nraneffs*nlevels)

    # Get number of unique random effects
    q_u = np.sum(nraneffs*(nraneffs+1)//2)
    
    # Get number of parameters
    L1 = str2vec(inputs['contrasts'][0]['c' + str(1)]['vector'])
    L1 = np.array(L1)
    p = L1.shape[0]
    del L1, rfxdes, rfxfac
    
    # Read in the nifti size and work out number of voxels.
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = loadFile(nifti_path)

    NIFTIsize = nifti.shape
    v = int(np.prod(NIFTIsize))

    # --------------------------------------------------------------------------------
    # Get n (number of observations) and n_sv (spatially varying number of
    # observations)
    # --------------------------------------------------------------------------------

    # Work out number of batchs
    n_b = len(glob.glob(os.path.join(OutDir,"tmp","blmm_vox_n_batch*")))

    # Read in n (spatially varying)
    nmapb  = loadFile(os.path.join(OutDir,"tmp", "blmm_vox_n_batch1.nii"))
    n_sv = nmapb.get_data()# Read in uniqueness Mask file

    # Cycle through batches and add together n.
    for batchNo in range(2,(n_b+1)):
        
        # Obtain the full nmap.
        n_sv = n_sv + loadFile(os.path.join(OutDir,"tmp", 
            "blmm_vox_n_batch" + str(batchNo) + ".nii")).get_data()
        
    # Save nmap
    nmap = nib.Nifti1Image(n_sv,
                           nifti.affine,
                           header=nifti.header)
    nib.save(nmap, os.path.join(OutDir,'blmm_vox_n.nii')) ### MARKER: THIS SHOULD ONLY BE DONE ONCE BUT CURRENTLY DONE IN ALL BATCHES
    n_sv = n_sv.reshape(v, 1)
    del nmap

    # Get ns.
    X = loadFile(inputs['X'])
    n = X.shape[0]

    # --------------------------------------------------------------------------------
    # Create Mask ### MARKER: THIS SHOULD ONLY BE DONE ONCE BUT CURRENTLY DONE IN ALL BATCHES
    # --------------------------------------------------------------------------------

    Mask = np.ones([v, 1])

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
            Mask[n_sv<rmThresh*n]=0

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
            Mask[n_sv<amThresh]=0

    # We remove anything with 1 degree of freedom (or less) by default.
    # 1 degree of freedom seems to cause broadcasting errors on a very
    # small percentage of voxels.
    Mask[n_sv<=p+1]=0

    if 'analysis_mask' in inputs:

        amask_path = inputs["analysis_mask"]
        
        # Read in the mask nifti.
        amask = loadFile(amask_path).get_data().reshape([v,1])

    else:

        # By default make amask ones
        amask = np.ones([v,1])


    # Get indices for whole analysis mask. These indices are the indices we
    # have recorded for the product matrices with respect to the entire volume
    amInds = get_amInds(amask)
        
    # Ensure overall mask matches analysis mask
    Mask[~np.in1d(np.arange(v).reshape(v,1), amInds)]=0

    # Output final mask map
    maskmap = nib.Nifti1Image(Mask.reshape(
                                    NIFTIsize[0],
                                    NIFTIsize[1],
                                    NIFTIsize[2]
                                    ),
                              nifti.affine,
                              header=nifti.header) ### MARKER: THIS SHOULD ONLY BE DONE ONCE BUT CURRENTLY DONE IN ALL BATCHES
    nib.save(maskmap, os.path.join(OutDir,'blmm_vox_mask.nii'))
    del maskmap

    # ------------------------------------------------------------------------
    # Work out block of voxels we are looking at
    # ------------------------------------------------------------------------
    # Get indices for block. These indices have to be the indices we want to
    # compute, in relation to the entire volume. If we aren't partitioning by 
    # block these will be equal to amInds
    pnvb = pracNumVoxelBlocks(inputs)
    bamInds = get_amInds(amask, vb-1, pnvb) # Remem vb 0 indexed in py but 1 indexed in bash

    # Get indices of voxels in ring around brain where there are
    # missing studies.
    R_inds = np.sort(np.where((Mask_cv==1)*(n_sv<n))[0])

    # Work out the 'ring' indices, in relation to the analysis mask
    ix_r = np.argsort(np.argsort(R_inds))
    R_inds_am = np.sort(np.where(np.in1d(amInds,R_inds))[0])[ix_r]

    # Get indices of the "inner" volume where all studies had information
    # present. I.e. the voxels (usually near the middle of the brain) where
    # every voxel has a reading for every study.
    I_inds = np.sort(np.where((Mask_cv==1)*(n_sv==n))[0])

    # Work out the 'inner' indices, in relation to the analysis mask
    ix_i = np.argsort(np.argsort(I_inds))
    I_inds_am = np.sort(np.where(np.in1d(amInds,I_inds))[0])[ix_i]

    # ------------------------------------------------------------------------
    # Number of voxels in ring and inner
    # ------------------------------------------------------------------------

    # Number of voxels in ring
    v_r = R_inds.shape[0]

    # Number of voxels in inner mask
    v_i = I_inds.shape[0]

    # Number of voxels in whole (inner + ring) mask
    v_m = v_i + v_r

    # ------------------------------------------------------------------------
    # Degrees of freedom (n-p)
    # ------------------------------------------------------------------------

    # Create df map
    df_r = n_sv[R_inds,:] - p
    df_r = df_r.reshape([v_r])
    df_i = n - p

    # Unmask df
    df = np.zeros([v])
    df[R_inds] = df_r 
    df[I_inds] = df_i

    df = df.reshape(int(NIFTIsize[0]),
                    int(NIFTIsize[1]),
                    int(NIFTIsize[2]))

    # Save beta map.
    dfmap = nib.Nifti1Image(df,
                            nifti.affine,
                            header=nifti.header) ### MARKER: THIS SHOULD ONLY BE DONE ONCE BUT CURRENTLY DONE IN ALL BATCHES
    nib.save(dfmap, os.path.join(OutDir,'blmm_vox_edf.nii'))
    del df, dfmap

    # --------------------------------------------------------------------------------
    # Load X'X, X'Y, Y'Y, X'Z, Y'Z, Z'Z
    # --------------------------------------------------------------------------------

    # Ring X'Y, Y'Y, Z'Y
    XtY = readAndSumAtB('XtY',OutDir,np.arange(np.prod(amInds.shape)),n_b).reshape([v_r, p, 1])
    YtY = readAndSumAtB('YtY',OutDir,np.arange(np.prod(amInds.shape)),n_b).reshape([v_r, 1, 1])
    ZtY = readAndSumAtB('ZtY',OutDir,np.arange(np.prod(amInds.shape)),n_b).reshape([v_r, q, 1])

    # Remove X'Y, Y'Y and Z'Y files here
    for batchNo in range(1,(n_b+1)):
        os.remove(os.path.join(OutDir, "tmp","XtY" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp","YtY" + str(batchNo) + ".npy"))
        os.remove(os.path.join(OutDir, "tmp","ZtY" + str(batchNo) + ".npy"))

    # Save new X'Y, Y'Y and Z'Y files here
    np.save(os.path.join(OutDir,"tmp","YtY" + str(batchNo)), 
               YtY)
    np.save(os.path.join(OutDir,"tmp","XtY" + str(batchNo)), 
               XtY) 
    np.save(os.path.join(OutDir,"tmp","ZtY" + str(batchNo)), 
               ZtY) 

    # Ring Z'Z. Z'X, X'X
    if v_r:

        ZtZ_r = readAndSumUniqueAtB('ZtZ',OutDir,R_inds,n_b,True)
        ZtX_r = readAndSumUniqueAtB('ZtX',OutDir,R_inds,n_b,True)
        XtX_r = readAndSumUniqueAtB('XtX',OutDir,R_inds,n_b,True)

        print(ZtZ_r.shape)

        # Then work out new unique X'X, Z'X, Z'Z indices 
        # Note: finding the unique elements may change the order
        # so extra care must be taken here
        _, idx = np.unique(ZtZ_r, axis=0, return_index=True)
        ZtZ_ru = ZtZ_r[np.sort(idx),:]

        _, idx = np.unique(ZtX_r, axis=0, return_index=True)
        ZtX_ru = ZtX_r[np.sort(idx),:]

        _, idx = np.unique(ZtX_r, axis=0, return_index=True)
        XtX_ru = XtX_r[np.sort(idx),:]

        # Work out the uniqueness indices for the ring (the key by which
        # we recover X'X, X'Z, Z'Z etc). Note: Due to the preserving of
        # order above, these indices should be the same for X'X, Z'X and 
        # Z'Z, so we need only compute them once.
        ZtZ_r_df = pd.DataFrame(ZtZ_r)
        ZtZ_r_df['id'] = ZtZ_r_df.groupby(ZtZ_r_df.columns.tolist(), sort=False).ngroup() + 1
        unique_id_r = ZtZ_r_df['id'].values

    
    if v_i:
            
        # Inner Z'Z. Z'X, X'X
        ZtZ_i = readAndSumUniqueAtB('ZtZ',OutDir,I_inds,n_b,False)#.reshape([1, q, q])
        ZtX_i = readAndSumUniqueAtB('ZtX',OutDir,I_inds,n_b,False)#.reshape([1, q, p])
        XtX_i = readAndSumUniqueAtB('XtX',OutDir,I_inds,n_b,False)#.reshape([1, p, p])

        # Add to the list of unique designs
        ZtZ_u = np.concatenate((ZtZ_ru, ZtZ_i), axis=0)
        XtX_u = np.concatenate((XtX_ru, XtX_i), axis=0)
        ZtX_u = np.concatenate((ZtX_ru, ZtX_i), axis=0)

        # The unique id for the inner will be the next available value
        unique_id_i = np.max(unique_id_r)+1

    # Unmask uniqueness map
    uMap = np.zeros([v])
    uMap[R_inds] = unique_id_r 
    uMap[I_inds] = unique_id_i

    uMap = uMap.reshape(int(NIFTIsize[0]),
                        int(NIFTIsize[1]),
                        int(NIFTIsize[2]))

    # Save beta map.
    uMap = nib.Nifti1Image(uMap,
                           nifti.affine,
                           header=nifti.header) ### MARKER: THIS SHOULD ONLY BE DONE ONCE BUT CURRENTLY DONE IN ALL BATCHES
    nib.save(uMap, os.path.join(OutDir,'blmm_vox_uniqueM.nii'))

    # Save unique designs
    np.save(os.path.join(OutDir,"tmp","XtX" + str(batchNo)), 
               XtX_u)
    np.save(os.path.join(OutDir,"tmp","ZtX" + str(batchNo)), 
               ZtX_u) 
    np.save(os.path.join(OutDir,"tmp","ZtZ" + str(batchNo)), 
               ZtZ_u) 

    del uMap, ZtZ_u, XtX_u, ZtX_u

    w.resetwarnings()

# MARKER
def readAndSumAtB(AtBstr, OutDir, vinds,n_b):

    # Read in first A'B
    AtB = readLinesFromNPY(os.path.join(OutDir,"tmp",AtBstr + '1.npy'), vinds)

    # Cycle through batches and add together results.
    for batchNo in range(2,(n_b+1)):

        # Sum A'B
        AtB = AtB + readLinesFromNPY(os.path.join(OutDir,"tmp",AtBstr + str(batchNo) + ".npy"), vinds)

    # Return A'B
    return(AtB)

# MARKER
def readAndSumUniqueAtB(AtBstr, OutDir, vinds, n_b, sv):

    # Work out the uniqueness mask for the spatially varying designs
    uniquenessMask = loadFile(os.path.join(OutDir,"tmp", 
        "blmm_vox_uniqueM_batch1.nii")).get_data()

    v = np.prod(uniquenessMask.shape)
    vcurrent = np.prod(vinds.shape)

    uniquenessMask=uniquenessMask.reshape(v)

    # Work out how many unique matrices there were
    maxM = np.int32(np.amax(uniquenessMask))

    if sv:
        # Work out the uniqueness mask inside the ring around the brain
        uniquenessMask = uniquenessMask[vinds]
    else:
        # Work out the uniqueness mask value inside the inner part of the brain
        uniquenessMask = uniquenessMask[vinds[0]] 


    # read in XtX
    AtB_batch_unique = np.load(
        os.path.join(OutDir,"tmp",AtBstr+"1.npy"))

    # Make zeros for outer ring of brain ZtZ, XtX, ZtX etc (remember A'B is still flattened)
    if sv:
        AtB = np.zeros((vcurrent, AtB_batch_unique.shape[1]))

    # Fill with unique maskings
    for m in range(1,maxM+1):

        if sv:
            # Work out Z'Z, Z'X and X'X for the ring
            AtB[np.where(uniquenessMask==m),:] = AtB_batch_unique[(m-1),:]

        # Work out Z'Z, Z'X and X'X for the inner
        else:
            if uniquenessMask == m:
                AtB = AtB_batch_unique[(m-1),:]

    # Cycle through batches and add together results.
    for batchNo in range(2,(n_b+1)):

        # Read in uniqueness Mask file
        uniquenessMask = loadFile(os.path.join(OutDir,"tmp", 
            "blmm_vox_uniqueM_batch" + str(batchNo) + ".nii")).get_data().reshape(v)

        maxM = np.int32(np.amax(uniquenessMask))

        if sv:
            # Work out the uniqueness mask inside the ring around the brain
            uniquenessMask = uniquenessMask[vinds] 
        else:
            # Work out the uniqueness mask value inside the inner part of the brain
            uniquenessMask = uniquenessMask[vinds[0]] 


        # read in XtX, ZtX, ZtZ
        AtB_batch_unique = np.load(
            os.path.join(OutDir,"tmp",AtBstr + str(batchNo) + ".npy"))

        # Make zeros for whole nifti ZtZ, XtX, ZtX etc
        if sv:
            AtB_batch = np.zeros((vcurrent, AtB_batch_unique.shape[1]))

        # Fill with unique maskings
        for m in range(1,maxM+1):

            if sv:
                AtB_batch[np.where(uniquenessMask==m),:] = AtB_batch_unique[(m-1),:]
            else:
                # Work out Z'Z, Z'X and X'X for the inner
                if uniquenessMask == m:

                    AtB_batch = AtB_batch_unique[(m-1),:]

        # Add to running total
        AtB = AtB + AtB_batch

    return(AtB)


if __name__ == "__main__":
    main()
