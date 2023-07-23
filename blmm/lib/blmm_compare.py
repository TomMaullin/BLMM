import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import sys
import os
import yaml
from scipy import stats
np.set_printoptions(threshold=sys.maxsize)
from blmm.src.npMatrix3d import *
from blmm.src.fileio import *


# ====================================================================================
#
# This file is an additional optional stage for the BLMM pipeline. It provides code 
# for model comparison in the form of a chi^2 50:50 mixture Likelihood Ratio Test.
#
# ------------------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited: 09/05/2021)
#
# ------------------------------------------------------------------------------------
#
# The code takes the following inputs:
#
# ------------------------------------------------------------------------------------
#
# - `blmmDir1`: The output directory for the BLMM analysis which was the restricted 
#               model.
# - `blmmDir2`: The output directory for the BLMM analysis which was the full model. 
# - `OutDir`: Output directory.
#
# ====================================================================================
def compare(blmmDir1, blmmDir2, OutDir):

    # --------------------------------------------------------------------------------
    # Load in basic inputs
    # --------------------------------------------------------------------------------

    # A NIFTI for reference
    NIFTI = nib.load(os.path.join(blmmDir1, 'blm_vox_mask.nii'))

    # NIFTI features
    dim = (*NIFTI.shape,1)
    aff = NIFTI.affine
    hdr = NIFTI.header

    # Inputs for first directory.
    with open(os.path.join(blmmDir1, 'inputs.yml'), 'r') as stream:
        inputs1 = yaml.load(stream,Loader=yaml.FullLoader)

    # Inputs for second directory.
    with open(os.path.join(blmmDir2, 'inputs.yml'), 'r') as stream:
        inputs2 = yaml.load(stream,Loader=yaml.FullLoader)

    # --------------------------------------------------------------------------------
    # Create OutDir and remove previous inputs
    # --------------------------------------------------------------------------------

    # Make output directory and tmp
    if not os.path.isdir(OutDir):
        os.mkdir(OutDir)

    # If previous files exist, delete them.
    if os.path.exists(os.path.join(OutDir, 'blmm_vox_mask.nii')):
        os.remove(os.path.join(OutDir, 'blmm_vox_mask.nii'))    

    if os.path.exists(os.path.join(OutDir, 'blmm_vox_Chi2.nii')):
        os.remove(os.path.join(OutDir, 'blmm_vox_Chi2.nii'))

    if os.path.exists(os.path.join(OutDir, 'blmm_vox_Chi2lp.nii')):
        os.remove(os.path.join(OutDir, 'blmm_vox_Chi2lp.nii'))      

    # --------------------------------------------------------------------------------
    # Check if BLMM or BLM
    # --------------------------------------------------------------------------------

    # Check first model is blm or blmm
    if os.path.isfile(os.path.join(blmmDir1, 'blmm_vox_mask.nii')):

        # We have random effects in the model
        model1_isBLM = False

    elif os.path.isfile(os.path.join(blmmDir1, 'blm_vox_mask.nii')):

        # There are no random effects in the model
        model1_isBLM = True

    else:
        
        raise ValueError('Directory for model 1 is not a BLMM/BLM directory. Missing mask file.')

    # Check second model is blm or blmm
    if os.path.isfile(os.path.join(blmmDir2, 'blmm_vox_mask.nii')):

        # We have random effects in the model
        model2_isBLM = False

    elif os.path.isfile(os.path.join(blmmDir2, 'blm_vox_mask.nii')):

        # There are no random effects in the model
        model2_isBLM = True

    else:
        
        raise ValueError('Directory for model 2 is not a BLMM/BLM directory. Missing mask file.')

    # --------------------------------------------------------------------------------
    # Get q (number of random effects)
    # --------------------------------------------------------------------------------

    # If it's a BLM model there are no random effects
    if model1_isBLM:

        # No random effects
        q_des1 = 0
        qu_des1 = 0

        # No random factors
        r_des1 = 0

    else:

        # Random factor variables.
        rfxmats = inputs1['Z']

        # Number of random factors
        r_des1 = len(rfxmats)

        if r_des1 > 1:
            
            raise ValueError('Likelihood ratio testing is not supported for multifactor models.' +
                             ' Model 1 contains ' + str(r_des1) + '>1 factors.')

        # Number of random effects and levels
        nraneffs_des1 = []
        nlevels_des1 = []

        # Loop through each factor
        for k in range(r_des1):

            rfxdes = loadFile(rfxmats[k]['f' + str(k+1)]['design'])
            rfxfac = loadFile(rfxmats[k]['f' + str(k+1)]['factor'])

            nraneffs_des1 = nraneffs_des1 + [rfxdes.shape[1]]
            nlevels_des1 = nlevels_des1 + [len(np.unique(rfxfac))]

        # Get number of random effects
        nraneffs_des1 = np.array(nraneffs_des1)
        nlevels_des1 = np.array(nlevels_des1)
        q_des1 = np.sum(nraneffs_des1*nlevels_des1)

        # Number of unique random effects
        qu_des1 = np.sum(nraneffs_des1)

    # If it's a BLM model there are no random effects
    if model2_isBLM:

        # No random effects
        q_des2 = 0
        qu_des2 = 0

        # No random factors
        r_des2 = 0

    else:

        # Random factor variables.
        rfxmats = inputs2['Z']

        # Number of random factors
        r_des2 = len(rfxmats)

        if r_des2 > 1:
            
            raise ValueError('Likelihood ratio testing is not supported for multifactor models.' +
                             ' Model 1 contains ' + str(r_des2) + '>1 factors.')

        # Number of random effects and levels
        nraneffs_des2 = []
        nlevels_des2 = []

        # Loop through each factor
        for k in range(r_des2):

            rfxdes = loadFile(rfxmats[k]['f' + str(k+1)]['design'])
            rfxfac = loadFile(rfxmats[k]['f' + str(k+1)]['factor'])

            nraneffs_des2 = nraneffs_des2 + [rfxdes.shape[1]]
            nlevels_des2 = nlevels_des2 + [len(np.unique(rfxfac))]

        # Get number of random effects
        nraneffs_des2 = np.array(nraneffs_des2)
        nlevels_des2 = np.array(nlevels_des2)
        q_des2 = np.sum(nraneffs_des2*nlevels_des2)

        # Number of unique random effects
        qu_des2 = np.sum(nraneffs_des2)

    # --------------------------------------------------------------------------------
    # Check designs are correct way around
    # --------------------------------------------------------------------------------

    # If the first design is not smaller
    if qu_des1 > qu_des2:

        # Warn the user
        w.warn("Warning: Models were entered in the incorrect order and have been switched. " + \
               "Model 1 should be the restricted model but was instead the full model.")

        # Make temporary copy of model 2
        model1_isBLM_tmp = model2_isBLM
        blmmDir1_tmp = str(blmmDir2)
        inputs1_tmp = dict(inputs2)
        r_des1_tmp = r_des2
        nraneffs_des1_tmp = np.array(nraneffs_des2)
        nlevels_des1_tmp = np.array(nlevels_des2)
        q_des1_tmp = q_des2
        qu_des1_tmp = qu_des2

        # Set model 2 to model 1
        model2_isBLM = model1_isBLM
        blmmDir2 = str(blmmDir1)
        inputs2 = dict(inputs1)
        r_des2 = r_des1
        nraneffs_des2 = np.array(nraneffs_des1)
        nlevels_des2 = np.array(nlevels_des1)
        q_des2 = q_des1
        qu_des2 = qu_des1

        # Set model 1 to model 2 copy 
        model1_isBLM = model1_isBLM_tmp
        blmmDir1 = str(blmmDir1_tmp)
        inputs1 = dict(inputs1_tmp)
        r_des1 = r_des1_tmp
        nraneffs_des1 = np.array(nraneffs_des1_tmp)
        nlevels_des1 = np.array(nlevels_des1_tmp)
        q_des1 = q_des1_tmp
        qu_des1 = qu_des1_tmp

    # --------------------------------------------------------------------------------
    # Check the full model has only one more random effect than the reduced.
    # --------------------------------------------------------------------------------

    # Check if models are correctly nested
    if (qu_des1 != (qu_des2-1)):

        raise ValueError('Chi squared LRT testing is only supported for models which' + \
                         ' differ by one random effect. You have run an LRT for models ' + \
                         'with ' + str(qu_des1) + ' and ' + str(qu_des2) + ' random effects. ' + \
                         str(qu_des1) + ' does not equal ' + str(qu_des2) + '-1.')



    # --------------------------------------------------------------------------------
    # Sanity check same number of fixed effects
    # --------------------------------------------------------------------------------

    # Read in X for model 1.
    X_des1 = loadFile(inputs1['X'])

    # Work out p for model 1.
    p1 = X_des1.shape[-1]

    # Read in X for model 2.
    X_des2 = loadFile(inputs2['X'])

    # Work out p for model 2.
    p2 = X_des2.shape[-1]

    # Check if same number of fixed effects
    if (p1 != p2):

        raise ValueError('Both models must have the same fixed effects as parameter estimation' + \
                         ' is performed using restricted maximum likelihood estimation. However, ' + \
                         'models entered have ' + str(p1) + ' and ' + str(p2) + ' fixed effects.' + \
                         str(p1) + ' does not equal ' + str(p2) + '.')

    # --------------------------------------------------------------------------------
    # Get mask
    # --------------------------------------------------------------------------------

    # Load masks
    mask1 = nib.load(os.path.join(blmmDir1, 'blm_vox_mask.nii')).get_fdata()>0
    mask2 = nib.load(os.path.join(blmmDir2, 'blmm_vox_mask.nii')).get_fdata()>0

    # Combine masks
    mask = mask1*mask2

    # Save mask
    addBlockToNifti(os.path.join(OutDir, "blmm_vox_mask.nii"), mask.reshape(np.prod(mask.shape)),np.arange(np.prod(mask.shape)), volInd=0,dim=dim,aff=aff,hdr=hdr)

    # --------------------------------------------------------------------------------
    # Create and save Chi^2 image
    # --------------------------------------------------------------------------------

    # Load log likelihoods
    llh1 = nib.load(os.path.join(blmmDir1, 'blm_vox_llh.nii')).get_fdata()
    llh2 = nib.load(os.path.join(blmmDir2, 'blmm_vox_llh.nii')).get_fdata()

    # X^2 statistic
    Chi2 = np.maximum(-2*(llh1-llh2),0)

    # Save X^2 statistic
    addBlockToNifti(os.path.join(OutDir, "blmm_vox_Chi2.nii"), Chi2.reshape(np.prod(Chi2.shape)),np.arange(np.prod(Chi2.shape)), volInd=0,dim=dim,aff=aff,hdr=hdr)
    
    # --------------------------------------------------------------------------------
    # Create P-value image
    # --------------------------------------------------------------------------------

    # Work out the 50:50 CDF probabilities, if BLM this is just a point mass.
    if model1_isBLM:

        # Point mass cdf
        p = 0.5*(Chi2[mask]>=0)

    else:

        # Chi square CDF (sf has higher precision than cdf in scipy)
        p = 0.5*(1-stats.chi2.sf(Chi2[mask], int(qu_des1)))

    # Add the 50:50 CDF probability for the second model (cannot be BLM as then reduced
    # model would have to somehow have less)
    p = p + 0.5*(1-stats.chi2.sf(Chi2[mask], int(qu_des2)))

    # Transform to -10log(p)
    p = -np.log10(1-p)

    # Check if minlog given for model 1
    if ("minlog" in inputs1):
        minlog1=inputs1['minlog']
    else:
        minlog1=-323.3062153431158

    # Check if minlog given for model 2
    if ("minlog" in inputs2):
        minlog2=inputs2['minlog']
    else:
        minlog2=-323.3062153431158

    # Work out minlog
    minlog = np.minimum(minlog1,minlog2)

    # Remove infs
    p[np.logical_and(np.isinf(p), p<0)]=minlog

    # Make a p value volume (1 to 0)
    pvol = np.zeros(NIFTI.shape)
    pvol[mask]=p.reshape(pvol[mask].shape)

    # Save X^2 p-values
    addBlockToNifti(os.path.join(OutDir, "blmm_vox_Chi2lp.nii"), pvol.reshape(np.prod(pvol.shape)),np.arange(np.prod(pvol.shape)), volInd=0,dim=dim,aff=aff,hdr=hdr)


    # TODO OPTIONAL AICS?