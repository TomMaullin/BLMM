import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import scipy
import scipy.sparse
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
from scipy import ndimage
import time
import pandas as pd
from lib.fileio import *

# R preprocessing
def Rpreproc(OutDir,nvg,cv):

    # There should be an inputs file
    with open(os.path.join(OutDir,'inputs.yml'), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # -------------------------------------------------------------------------------------------------
    # Get Y
    # -------------------------------------------------------------------------------------------------

    # Y volumes
    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load one Y for reference
    Y0 = loadFile(Y_files[0])

    # -------------------------------------------------------------------------------------------------
    # Dim
    # -------------------------------------------------------------------------------------------------

    # Make sure in numpy format
    dim = np.array(Y0.shape)

    # Number of voxels
    v = np.prod(dim)

    # -------------------------------------------------------------------------------------------------
    # Number of observations
    # -------------------------------------------------------------------------------------------------
    # Number of observations
    X = pd.io.parsers.read_csv(inputs['X'], header=None).values
    n = X.shape[0]

    # -------------------------------------------------------------------------------------------------
    # Thresholding
    # -------------------------------------------------------------------------------------------------

    # Relative masking threshold
    rmThresh = inputs['Missingness']['MinPercent']

    # -------------------------------------------------------------------------------------------------
    # Voxel groups
    # -------------------------------------------------------------------------------------------------

    # Split voxels we want to look at into groups we can compute
    voxelGroups = np.array_split(np.arange(v), nvg)

    # Current group of voxels
    inds_cv = voxelGroups[cv]

    # Number of voxels currently (should be ~1000)
    v_current = len(inds_cv)

    # -------------------------------------------------------------------------------------------------
    # Get Masking
    # -------------------------------------------------------------------------------------------------

    # Mask volumes (if they are given)
    if 'data_mask_files' in inputs:

        # Rrad in mask files, making sure to avoid the newline characters
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

            # If we haven't the same number of masks and observations then
            # something must be mispecified.
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

    # Mask volumes (if they are given)
    if 'analysis_mask' in inputs:

        # Load the file and check it's shape is 3d (as oppose to 4d with a 4th dimension
        # of 1)
        M_a = loadFile(inputs['analysis_mask']).get_data()
        M_a = M_a.reshape((M_a.shape[0],M_a.shape[1],M_a.shape[2]))

    else:

        # Else set to None
        M_a = None

    # --------------------------------------------------------------------------------
    # Get q 
    # --------------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------------------------------
    # Read in Y and apply masking
    # -------------------------------------------------------------------------------------------------
    Y = np.zeros([n, v])
    for i in range(0, len(Y_files)):

        # Read in each individual NIFTI.
        Yi = loadFile(Y_files[i]).get_data()

        # Mask Y if necesary
        if M_files:
        
            # Apply mask
            M_indiv = loadFile(M_files[i]).get_data()
            Yi = np.multiply(Yi,M_indiv)

        # If theres an initial threshold for the data apply it.
        if M_t is not None:
            Yi[Yi<M_t]=0

        if M_a is not None:
            Yi[M_a==0]=0

        # NaN check
        Yi = np.nan_to_num(Yi)

        # Flatten Yi
        Yi = Yi.reshape(v)

        # Get just the voxels we're interested in
        Yi = Yi[inds_cv].reshape(1, v_current)

        # Concatenate
        if i==0:
            Y_concat = Yi
        else:
            Y_concat = np.concatenate((Y_concat, Yi), axis=0)

    # Loop through voxels checking missingness
    for vox in np.arange(v_current):

        # Threshold out the voxels which have too much missingness
        if np.count_nonzero(Y_concat[:,vox], axis=0)/n < rmThresh:

            # If we don't have enough data lets replace that voxel 
            # with zeros
            Y_concat[:,vox] = np.zeros(Y_concat[:,vox].shape)

        # Threshold out the voxels which are underidentified (same
        # practice as lmer)
        if np.count_nonzero(Y_concat[:,vox], axis=0) <= q:

            # If we don't have enough data lets replace that voxel 
            # with zeros
            Y_concat[:,vox] = np.zeros(Y_concat[:,vox].shape)

    # Write out Z in full to a csv file
    pd.DataFrame(Y_concat.reshape(n,v_current)).to_csv(os.path.join(OutDir,"data","Y_Rversion_" + str(cv) + ".csv"), header=None, index=None)


