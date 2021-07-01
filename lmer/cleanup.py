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
from matplotlib import pyplot as plt
from statsmodels.stats import multitest


# ===========================================================================
#
# Inputs:
#
# ---------------------------------------------------------------------------
#
# ===========================================================================
def cleanup(OutDir):

    # -----------------------------------------------------------------------
    # Create results directory (if we are on the first simulation)
    # -----------------------------------------------------------------------
    # Results directory
    resDir = os.path.join(OutDir,'results')

    # If resDir doesn't exist, make it
    if not os.path.exists(resDir):
        os.mkdir(resDir)

    # -----------------------------------------------------------------------
    # N map
    # -----------------------------------------------------------------------

    # Get spatially varying n
    n_sv = nib.load(os.path.join(OutDir, 'blmm_vox_n.nii')).get_data()

    # Work out number of subjects
    n = np.amax(n_sv)

    # Work out which voxels had readings for all subjects
    loc_sv = (n_sv>n//2)&(n_sv<n)
    loc_nsv = (n_sv==n)

    # Work out number of spatially varying and non-spatially varying voxels
    v_sv = np.sum(loc_sv)
    v_nsv = np.sum(loc_nsv)

    # Make line to add to csv for MRD
    n_line = np.array([[v_sv + v_nsv, v_sv, v_nsv]])

    # Number of voxels
    fname_n = os.path.join(resDir, 'n.csv')

    # Add to files 
    addLineToCSV(fname_n, n_line)

    # -----------------------------------------------------------------------
    # MAE and MRD for beta maps
    # -----------------------------------------------------------------------

    # Get BLMM beta
    beta_blmm = nib.load(os.path.join(OutDir, 'BLMM', 'blmm_vox_beta.nii')).get_data()

    # Get lmer beta
    beta_lmer = nib.load(os.path.join(OutDir, 'lmer', 'lmer_vox_beta.nii')).get_data()

    # Remove zero values (spatially varying)
    beta_blmm_sv = beta_blmm[(beta_lmer!=0) & loc_sv]
    beta_lmer_sv = beta_lmer[(beta_lmer!=0) & loc_sv]

    # Remove zero values (non spatially varying)
    beta_blmm_nsv = beta_blmm[(beta_lmer!=0) & loc_nsv]
    beta_lmer_nsv = beta_lmer[(beta_lmer!=0) & loc_nsv]

    # Remove zero values (both)
    beta_blmm = beta_blmm[beta_lmer!=0]
    beta_lmer = beta_lmer[beta_lmer!=0]

    # Get MAE
    MAE_beta = np.mean(np.abs(beta_blmm-beta_lmer))
    MAE_beta_sv = np.mean(np.abs(beta_blmm_sv-beta_lmer_sv))
    MAE_beta_nsv = np.mean(np.abs(beta_blmm_nsv-beta_lmer_nsv))

    # Get MRD
    MRD_beta = np.mean(2*np.abs((beta_blmm-beta_lmer)/(beta_blmm+beta_lmer)))
    MRD_beta_sv = np.mean(2*np.abs((beta_blmm_sv-beta_lmer_sv)/(beta_blmm_sv+beta_lmer_sv)))
    MRD_beta_nsv = np.mean(2*np.abs((beta_blmm_nsv-beta_lmer_nsv)/(beta_blmm_nsv+beta_lmer_nsv)))

    # Make line to add to csv for MAE
    MAE_beta_line = np.array([[MAE_beta, MAE_beta_sv, MAE_beta_nsv]])

    # Make line to add to csv for MRD
    MRD_beta_line = np.array([[MRD_beta, MRD_beta_sv, MRD_beta_nsv]])

    # MAE beta file name
    fname_MAE = os.path.join(resDir, 'MAE_beta.csv')

    # MRD beta file name
    fname_MRD = os.path.join(resDir, 'MRD_beta.csv')

    # Add to files 
    addLineToCSV(fname_MAE, MAE_beta_line)
    addLineToCSV(fname_MRD, MRD_beta_line)

    # Cleanup
    del beta_lmer, beta_blmm, MAE_beta, MRD_beta, MAE_beta_line, MRD_beta_line

    # -----------------------------------------------------------------------
    # MAE and MRD for sigma2 maps
    # -----------------------------------------------------------------------

    # Get BLMM sigma2
    sigma2_blmm = nib.load(os.path.join(OutDir, 'BLMM', 'blmm_vox_sigma2.nii')).get_data()

    # Get lmer sigma2
    sigma2_lmer = nib.load(os.path.join(OutDir, 'lmer', 'lmer_vox_sigma2.nii')).get_data()

    # Remove zero values (spatially varying)
    sigma2_blmm_sv = sigma2_blmm[(sigma2_lmer!=0) & loc_sv]
    sigma2_lmer_sv = sigma2_lmer[(sigma2_lmer!=0) & loc_sv]

    # Remove zero values (non spatially varying)
    sigma2_blmm_nsv = sigma2_blmm[(sigma2_lmer!=0) & loc_nsv]
    sigma2_lmer_nsv = sigma2_lmer[(sigma2_lmer!=0) & loc_nsv]

    # Remove zero values
    sigma2_blmm = sigma2_blmm[sigma2_lmer!=0]
    sigma2_lmer = sigma2_lmer[sigma2_lmer!=0]

    # Get MAE
    MAE_sigma2 = np.mean(np.abs(sigma2_blmm-sigma2_lmer))
    MAE_sigma2_sv = np.mean(np.abs(sigma2_blmm_sv-sigma2_lmer_sv))
    MAE_sigma2_nsv = np.mean(np.abs(sigma2_blmm_nsv-sigma2_lmer_nsv))

    # Get MRD
    MRD_sigma2 = np.mean(2*np.abs((sigma2_blmm-sigma2_lmer)/(sigma2_blmm+sigma2_lmer)))
    MRD_sigma2_sv = np.mean(2*np.abs((sigma2_blmm_sv-sigma2_lmer_sv)/(sigma2_blmm_sv+sigma2_lmer_sv)))
    MRD_sigma2_nsv = np.mean(2*np.abs((sigma2_blmm_nsv-sigma2_lmer_nsv)/(sigma2_blmm_nsv+sigma2_lmer_nsv)))

    # Make line to add to csv for MAE
    MAE_sigma2_line = np.array([[MAE_sigma2, MAE_sigma2_sv, MAE_sigma2_nsv]])

    # Make line to add to csv for MRD
    MRD_sigma2_line = np.array([[MRD_sigma2, MRD_sigma2_sv, MRD_sigma2_nsv]])

    # MAE sigma2 file name
    fname_MAE = os.path.join(resDir, 'MAE_sigma2.csv')

    # MRD sigma2 file name
    fname_MRD = os.path.join(resDir, 'MRD_sigma2.csv')

    # Add to files 
    addLineToCSV(fname_MAE, MAE_sigma2_line)
    addLineToCSV(fname_MRD, MRD_sigma2_line)

    # Cleanup
    del sigma2_lmer, sigma2_blmm, MAE_sigma2, MRD_sigma2, MAE_sigma2_line, MRD_sigma2_line

    # -----------------------------------------------------------------------
    # MAE and MRD for vechD maps
    # -----------------------------------------------------------------------

    # Get BLMM vechD
    vechD_blmm = nib.load(os.path.join(OutDir, 'BLMM', 'blmm_vox_D.nii')).get_data()

    # Get lmer vechD
    vechD_lmer = nib.load(os.path.join(OutDir, 'lmer', 'lmer_vox_D.nii')).get_data()

    # Remove zero values (Spatially varying)
    vechD_blmm_sv = vechD_blmm[(vechD_lmer!=0) & loc_sv]
    vechD_lmer_sv = vechD_lmer[(vechD_lmer!=0) & loc_sv]

    # Remove zero values (Non spatially varying)
    vechD_blmm_nsv = vechD_blmm[(vechD_lmer!=0) & loc_nsv]
    vechD_lmer_nsv = vechD_lmer[(vechD_lmer!=0) & loc_nsv]

    # Remove zero values
    vechD_blmm = vechD_blmm[vechD_lmer!=0]
    vechD_lmer = vechD_lmer[vechD_lmer!=0]

    # Get MAE
    MAE_vechD = np.mean(np.abs(vechD_blmm-vechD_lmer))
    MAE_vechD_sv = np.mean(np.abs(vechD_blmm_sv-vechD_lmer_sv))
    MAE_vechD_nsv = np.mean(np.abs(vechD_blmm_nsv-vechD_lmer_nsv))

    # Get MRD
    MRD_vechD = np.mean(2*np.abs((vechD_blmm-vechD_lmer)/(vechD_blmm+vechD_lmer)))
    MRD_vechD_sv = np.mean(2*np.abs((vechD_blmm_sv-vechD_lmer_sv)/(vechD_blmm_sv+vechD_lmer_sv)))
    MRD_vechD_nsv = np.mean(2*np.abs((vechD_blmm_nsv-vechD_lmer_nsv)/(vechD_blmm_nsv+vechD_lmer_nsv)))

    # Make line to add to csv for MAE
    MAE_vechD_line = np.array([[MAE_vechD, MAE_vechD_sv, MAE_vechD_nsv]])

    # Make line to add to csv for MRD
    MRD_vechD_line = np.array([[MRD_vechD, MRD_vechD_sv, MRD_vechD_nsv]])

    # MAE vechD file name
    fname_MAE = os.path.join(resDir, 'MAE_vechD.csv')

    # MRD vechD file name
    fname_MRD = os.path.join(resDir, 'MRD_vechD.csv')

    # Add to files 
    addLineToCSV(fname_MAE, MAE_vechD_line)
    addLineToCSV(fname_MRD, MRD_vechD_line)

    # Cleanup
    del vechD_lmer, vechD_blmm, MAE_vechD, MRD_vechD, MAE_vechD_line, MRD_vechD_line

    # -----------------------------------------------------------------------
    # Log-likelihood difference
    # -----------------------------------------------------------------------

    # Get BLMM llh
    llh_blmm = nib.load(os.path.join(OutDir, 'blmm_vox_llh.nii')).get_data()

    # Get lmer llh
    llh_lmer = nib.load(os.path.join(OutDir, 'lmer', 'lmer_vox_llh.nii')).get_data()

    # Remove zero values (spatially varying)
    llh_blmm_sv = llh_blmm[(llh_lmer!=0) & loc_sv]
    llh_lmer_sv = llh_lmer[(llh_lmer!=0) & loc_sv]

    # Remove zero values (non spatially varying)
    llh_blmm_nsv = llh_blmm[(llh_lmer!=0) & loc_nsv]
    llh_lmer_nsv = llh_lmer[(llh_lmer!=0) & loc_nsv]

    # Remove zero values
    llh_blmm = llh_blmm[llh_lmer!=0]
    llh_lmer = llh_lmer[llh_lmer!=0]

    # Get maximum absolute difference
    MAD_llh = np.mean(np.abs(llh_blmm-llh_lmer))
    MAD_llh_sv = np.mean(np.abs(llh_blmm_sv-llh_lmer_sv))
    MAD_llh_nsv = np.mean(np.abs(llh_blmm_nsv-llh_lmer_nsv))

    # Make line to add to csv for MAD
    MAD_llh_line = np.array([[MAD_llh, MAD_llh_sv, MAD_llh_nsv]])

    # MAD llh file name
    fname_MAD = os.path.join(resDir, 'MAD_llh.csv')

    # Add to files 
    addLineToCSV(fname_MAD, MAD_llh_line)

    # Cleanup
    del llh_lmer, llh_blmm, MAD_llh, MAD_llh_line

    
    # -----------------------------------------------------------------------
    # Times
    # -----------------------------------------------------------------------

    # Get BLMM times
    times_blmm = nib.load(os.path.join(OutDir, 'blmm_vox_times.nii')).get_data()

    # Get lmer times
    times_lmer = nib.load(os.path.join(OutDir, 'lmer', 'lmer_vox_times.nii')).get_data()

    # Remove zero values
    times_blmm_sv = times_blmm[(times_lmer!=0) & loc_sv]
    times_lmer_sv = times_lmer[(times_lmer!=0) & loc_sv]

    # Remove zero values
    times_blmm_nsv = times_blmm[(times_lmer!=0) & loc_nsv]
    times_lmer_nsv = times_lmer[(times_lmer!=0) & loc_nsv]

    # Remove zero values
    times_blmm = times_blmm[times_lmer!=0]
    times_lmer = times_lmer[times_lmer!=0]

    # Get mean difference
    MD_times_sv = np.mean(times_lmer_sv-times_blmm_sv)
    MD_times_nsv = np.mean(times_lmer_nsv-times_blmm_nsv)
    MD_times = np.mean(times_lmer-times_blmm)

    # Get total difference
    TD_times_sv = np.sum(times_lmer_sv-times_blmm_sv)
    TD_times_nsv = np.sum(times_lmer_nsv-times_blmm_nsv)
    TD_times = np.sum(times_lmer-times_blmm)

    # Get total lmer
    lmer_times_sv = np.sum(times_lmer_sv)
    lmer_times_nsv = np.sum(times_lmer_nsv)
    lmer_times = np.sum(times_lmer)

    # Get total blmm
    blmm_times_sv = np.sum(times_blmm_sv)
    blmm_times_nsv = np.sum(times_blmm_nsv)
    blmm_times = np.sum(times_blmm)

    # Make line to add to csv for MD
    MD_times_line = np.array([[MD_times, MD_times_sv, MD_times_nsv]])

    # Make line to add to csv for TD
    TD_times_line = np.array([[TD_times, TD_times_sv, TD_times_nsv]])

    # Make line to add to csv for lmer
    lmer_times_line = np.array([[lmer_times, lmer_times_sv, lmer_times_nsv]])

    # Make line to add to csv for blmm
    blmm_times_line = np.array([[blmm_times, blmm_times_sv, blmm_times_nsv]])

    # MD times file name
    fname_MD = os.path.join(resDir, 'MD_times.csv')

    # TD times file name
    fname_TD = os.path.join(resDir, 'TD_times.csv')

    # lmer times file name
    fname_lmer = os.path.join(resDir, 'lmer_times.csv')

    # blmm times file name
    fname_blmm = os.path.join(resDir, 'blmm_times.csv')

    # Add to files 
    addLineToCSV(fname_MD, MD_times_line)

    # Add to files 
    addLineToCSV(fname_TD, TD_times_line)

    # Add to files 
    addLineToCSV(fname_lmer, lmer_times_line)

    # Add to files 
    addLineToCSV(fname_blmm, blmm_times_line)

    # Cleanup
    del times_lmer, times_blmm, MD_times, MD_times_line, TD_times, TD_times_line


    # -----------------------------------------------------------------------
    # P values 
    # -----------------------------------------------------------------------
    # Load logp map
    logp = nib.load(os.path.join(OutDir, 'blmm_vox_conTlp.nii')).get_data()

    # Remove zeros
    logp = logp[logp!=0]

    # Un-"log"
    p = 10**(-logp)

    # Load logp map
    logp_lmer = nib.load(os.path.join(OutDir, 'lmer', 'lmer_vox_conTlp.nii')).get_data()

    # Remove zeros
    logp_lmer = logp_lmer[logp_lmer!=0]

    # Un-"log"
    p_lmer = 10**(-logp_lmer)

    # Get bin counts
    counts,_,_=plt.hist(p, bins=100, label='hist')

    # Make line to add to csv for bin counts
    pval_line = np.concatenate((np.array([counts])),axis=1)

    # pval file name
    fname_pval = os.path.join(resDir, 'pval_counts.csv')

    # Add to files 
    addLineToCSV(fname_pval, pval_line)

    # Get bin counts
    counts_lmer,_,_=plt.hist(p_lmer, bins=100, label='hist')

    # Make line to add to csv for bin counts
    pval_lmer_line = np.concatenate((np.array([counts_lmer])),axis=1)

    # pval file name
    fname_pval_lmer = os.path.join(resDir, 'pval_lmer_counts.csv')

    # Add to files 
    addLineToCSV(fname_pval_lmer, pval_lmer_line)

    # Convert to one tailed
    p_ot = np.zeros(p.shape)
    p_ot[p<0.5] = 2*p[p<0.5]
    p_ot[p>0.5] = 2*(1-p[p>0.5])
    p = p_ot

    # Convert to one tailed
    p_lmer_ot = np.zeros(p_lmer.shape)
    p_lmer_ot[p_lmer<0.5] = 2*p_lmer[p_lmer<0.5]
    p_lmer_ot[p_lmer>0.5] = 2*(1-p_lmer[p_lmer>0.5])
    p_lmer = p_lmer_ot

    # Perform bonferroni
    fwep_bonferroni = multitest.multipletests(p,alpha=0.05,method='bonferroni')[0]

    # Get number of false positives
    fwep_bonferroni = np.sum(fwep_bonferroni)

    # Perform bonferroni
    fwep_lmer_bonferroni = multitest.multipletests(p_lmer,alpha=0.05,method='bonferroni')[0]

    # Get number of false positives
    fwep_lmer_bonferroni = np.sum(fwep_lmer_bonferroni)

    # Make line to add to csv for fwe
    fwe_line = np.concatenate((np.array([[fwep_bonferroni]]),
                               np.array([[fwep_lmer_bonferroni]])),axis=1)

    # pval file name
    fname_fwe = os.path.join(resDir, 'pval_fwe.csv')

    # Add to files 
    addLineToCSV(fname_fwe, fwe_line)

    # Cleanup
    del p, logp, counts, fname_pval, pval_line, p_lmer, logp_lmer, counts_lmer, fname_pval_lmer, pval_lmer_line, fname_fwe, fwe_line

    # -----------------------------------------------------------------------
    # Cleanup finished!
    # -----------------------------------------------------------------------

    print('----------------------------------------------------------------')
    print('Cleanup complete!')
    print('----------------------------------------------------------------')


# Add R output to nifti files
def Rcleanup(OutDir, nvg, cv):

    # -----------------------------------------------------------------------
    # Read in design in BLMM inputs form (this just is easier as code already
    # exists for using this format).
    # -----------------------------------------------------------------------
    # There should be an inputs file in each simulation directory
    with open(os.path.join(OutDir,'inputs.yml'), 'r') as stream:
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

    # Number of covariance parameters
    ncov = np.sum(nraneffs*(nraneffs+1)//2)

    # -----------------------------------------------------------------------
    # Get number of observations and fixed effects
    # -----------------------------------------------------------------------
    X = pd.io.parsers.read_csv(inputs['X'], header=None).values
    n = X.shape[0]
    p = X.shape[1]

    # -----------------------------------------------------------------------
    # Get number voxels and dimensions
    # -----------------------------------------------------------------------

    # Y volumes
    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load one Y for reference
    Y0 = loadFile(Y_files[0])

    # Make sure in numpy format
    dim = np.array(Y0.shape)

    # Work out affine
    affine = Y0.affine.copy()

    # Number of voxels
    v = np.prod(dim)

    # Delete nmap
    del Y0
    
    # -------------------------------------------------------------------
    # Voxels of interest
    # -------------------------------------------------------------------

    # Split voxels we want to look at into groups we can compute
    voxelGroups = np.array_split(np.arange(v), nvg)

    # Current group of voxels
    inds_cv = voxelGroups[cv]

    # Number of voxels currently
    v_current = len(inds_cv)

    # -------------------------------------------------------------------
    # Beta combine
    # -------------------------------------------------------------------

    # Read in file
    beta_current = pd.io.parsers.read_csv(os.path.join(OutDir, 'lmer', 'beta_' + str(cv) + '.csv')).values

    print('beta_current shape', beta_current.shape)

    # Loop through parameters adding them one voxel at a time
    for param in np.arange(p):

        # Add back to a NIFTI file
        addBlockToNifti(os.path.join(OutDir,"lmer","lmer_vox_beta.nii"), beta_current[:,param], inds_cv, volInd=param,dim=(*dim,int(p)))

    # Remove file
    os.remove(os.path.join(OutDir, 'lmer', 'beta_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Sigma2 combine
    # -------------------------------------------------------------------

    # Read in file
    sigma2_current = pd.io.parsers.read_csv(os.path.join(OutDir, 'lmer', 'sigma2_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(OutDir,"lmer","lmer_vox_sigma2.nii"), sigma2_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(OutDir, 'lmer', 'sigma2_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # vechD combine
    # -------------------------------------------------------------------

    # Read in file
    vechD_current = pd.io.parsers.read_csv(os.path.join(OutDir, 'lmer', 'vechD_' + str(cv) + '.csv')).values

    # Loop through covariance parameters adding them one voxel at a time
    for param in np.arange(ncov):

        # Add back to a NIFTI file
        addBlockToNifti(os.path.join(OutDir,"lmer","lmer_vox_D.nii"), vechD_current[:,param], inds_cv, volInd=param,dim=(*dim,int(ncov)))

    # Remove file
    os.remove(os.path.join(OutDir, 'lmer', 'vechD_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Log-likelihood combine
    # -------------------------------------------------------------------

    # Read in file
    llh_current = pd.io.parsers.read_csv(os.path.join(OutDir, 'lmer', 'llh_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(OutDir,"lmer","lmer_vox_llh.nii"), llh_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(OutDir, 'lmer', 'llh_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # T statistic combine
    # -------------------------------------------------------------------

    # Read in file
    Tstat_current = pd.io.parsers.read_csv(os.path.join(OutDir, 'lmer', 'Tstat_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(OutDir,"lmer","lmer_vox_conT.nii"), Tstat_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(OutDir, 'lmer', 'Tstat_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # P value combine
    # -------------------------------------------------------------------

    # Read in file
    Pval_current = pd.io.parsers.read_csv(os.path.join(OutDir, 'lmer', 'Pval_' + str(cv) + '.csv')).values

    # Change to log scale
    Pval_current[Pval_current!=0]=-np.log10(Pval_current[Pval_current!=0])

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(OutDir,"lmer","lmer_vox_conTlp.nii"), Pval_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(OutDir, 'lmer', 'Pval_' + str(cv) + '.csv'))

    # -------------------------------------------------------------------
    # Times combine
    # -------------------------------------------------------------------

    # Read in file
    times_current = pd.io.parsers.read_csv(os.path.join(OutDir, 'lmer', 'times_' + str(cv) + '.csv')).values

    # Add back to a NIFTI file
    addBlockToNifti(os.path.join(OutDir,"lmer","lmer_vox_times.nii"), times_current, inds_cv, volInd=0,dim=(*dim,1))

    # Remove file
    os.remove(os.path.join(OutDir, 'lmer', 'times_' + str(cv) + '.csv'))

# This function adds a line to a csv. If the csv does not exist it creates it.
# It uses a filelock system
def addLineToCSV(fname, line):

    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            f = os.open(fname + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True

    # Check if file already exists and if so read it in
    if os.path.isfile(fname):

        # Read in data
        data = pd.io.parsers.read_csv(fname, header=None, index_col=None).values

        # Append line to data
        data = np.concatenate((data, line),axis=0)

    else:

        # The data is just this line
        data = line

    # Write data back to file
    pd.DataFrame(data).to_csv(fname, header=None, index=None)

    # Delete lock file, so other jobs know they can now write to the
    # file
    os.remove(fname + ".lock")
    os.close(f)

    del fname