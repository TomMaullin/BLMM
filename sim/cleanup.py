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


# ===========================================================================
#
# Inputs:
#
# ---------------------------------------------------------------------------
#
# ===========================================================================
def cleanup(OutDir,simNo):

    # -----------------------------------------------------------------------
    # Get simulation directory
    # -----------------------------------------------------------------------
    # Simulation directory
    simDir = os.path.join(OutDir, 'sim' + str(simNo))

    # -----------------------------------------------------------------------
    # Read in design in BLMM inputs form (this just is easier as code already
    # exists for using this format).
    # -----------------------------------------------------------------------
    # There should be an inputs file in each simulation directory
    with open(os.path.join(simDir,'inputs.yml'), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # -----------------------------------------------------------------------
    # Get number of random effects, levels and random factors in design
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Get number of observations and fixed effects
    # -----------------------------------------------------------------------
    X = pd.io.parsers.read_csv(os.path.join(simDir,"data","X.csv"), header=None).values
    n = X.shape[0]
    p = X.shape[1]

    print('n, p, q: ', n, p, q)

    # -----------------------------------------------------------------------
    # Get number voxels and dimensions
    # -----------------------------------------------------------------------

    # nmap location 
    nmap = os.path.join(simDir, "BLMM", "blmm_vox_n.nii")

    # Work out dim if we don't already have it
    dim = nib.Nifti1Image.from_filename(nmap, mmap=False).shape

    # Work out affine
    affine = nib.Nifti1Image.from_filename(nmap, mmap=False).affine.copy()

    # Number of voxels
    v = np.prod(dim)

    # Delete nmap
    del nmap

    print('v, dim ', v, dim)

    # -----------------------------------------------------------------------
    # Remove data directory
    # -----------------------------------------------------------------------
    shutil.rmtree(os.path.join(simDir, 'data'))

    # -----------------------------------------------------------------------
    # Convert R files to NIFTI images
    # -----------------------------------------------------------------------

    # List all beta parameter files for simulation
    beta_files = glob.glob(os.path.join(SimDir,'lmer', 'beta_*'))

    # Work out number of groups we have to split indices into.
    nvg = len(beta_files)
    print('nvg', nvg)
    
    # Split voxels we want to look at into groups we can compute
    voxelGroups = np.array_split(np.arange(v), nvg)

    # Loop through each file reading in one at a time and adding to nifti
    for cv, beta_file in enumerate(beta_files):

        # Current group of voxels
        inds_cv = voxelGroups[cv]

        # Number of voxels currently
        v_current = len(inds_cv)

        print('beta file: ', beta_file)

        # Read in file
        beta_current = pd.io.parsers.read_csv(filepath, header=None).values

        print('beta_current shape', beta_current.shape)

        # Loop through parameters adding them one voxel at a time
        for param in np.arange(p):

            # Add back to a NIFTI file
            addBlockToNifti(os.path.join(simDir,"lmer","lmer_beta.nii"), beta_current[:,param], inds_cv, volInd=param,dim=dim)






    # write.csv(betas,paste(lmerDir,'/beta_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
    # write.csv(sigma2,paste(lmerDir,'/sigma2_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
    # write.csv(vechD,paste(lmerDir,'/vechD_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
    # write.csv(llh,paste(lmerDir,'/llh_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
    # write.csv(tct,paste(lmerDir,'/time_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
    # write.csv(nvox_est,paste(lmerDir,'/v_est_',toString(batchNo),'.csv',sep=''), row.names = FALSE)




    # -----------------------------------------------------------------------
    # P value counts for histograms
    # -----------------------------------------------------------------------
    #plt.hist(nparray, bins=10, label='hist')