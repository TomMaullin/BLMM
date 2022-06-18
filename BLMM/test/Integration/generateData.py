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
from BLMM.lib.fileio import *
import time
import pandas as pd

# ===========================================================================
#
# Inputs:
#
# ---------------------------------------------------------------------------
#
# - `fwhm`: Full Width Half Maximum for noise smoothness. Must be given as an
#           np array. Can include 0 or None for dimensions not to be
#           smoothed.
# - `dim`: Dimensions of data to be generated. Must be given as an np array.
#
# ===========================================================================
def generate_data(n,dim,OutDir,testNo,desInd):

    # Make test directory
    testDir = os.path.join(OutDir, 'test' + str(testNo))
    if not os.path.exists(testDir):
        os.mkdir(testDir)

    # Make new data directory.
    if not os.path.exists(os.path.join(testDir,"data")):
        os.mkdir(os.path.join(testDir,"data"))

    # Make sure in numpy format (added 20 for smoothing)
    origdim = np.array(dim)
    dim = origdim + 20

    # -------------------------------------------------
    # Design parameters
    # -------------------------------------------------

    # Number of fixed effects parameters
    p = 4

    # Check which design we want
    if desInd == 3:

        # Number of random effects grouping factors
        r = 2

        # Number of levels for each factor
        nlevels = np.array([20,10])

        # Number of levels for each factor
        nraneffs = np.array([2,1])

    elif desInd == 2:

        # Number of random effects grouping factors
        r = 1

        # Number of levels for each factor
        nlevels = np.array([50])

        # Number of levels for each factor
        nraneffs = np.array([2])

    elif desInd == 1:

        # Number of random effects grouping factors
        r = 1

        # Number of levels for each factor
        nlevels = np.array([100])

        # Number of levels for each factor
        nraneffs = np.array([1])

    # fwhm for smoothing
    fwhm = 5

    # Number of voxels
    v = np.prod(dim)

    # Second dimension of Z
    q = np.sum(nlevels*nraneffs)

    # Relative missingness threshold (percentage)
    rmThresh = 0.5

    # -------------------------------------------------
    # Obtain design matrices
    # -------------------------------------------------

    # Fixed effects design matrix
    X = get_X(n, p)
    
    # Random effects design matrix, factor vectors and
    # raw regressor matrices (useful for generating
    # Y)
    Z, fDict, rrDict = get_Z(n, nlevels, nraneffs)

    # -------------------------------------------------
    # Obtain beta parameter vector
    # -------------------------------------------------

    # Get beta
    beta = get_beta(p)

    # -------------------------------------------------
    # Get Dhalf
    # -------------------------------------------------
    
    # New empty dictionary for Dhalf
    DhalfDict = dict()

    # Dhalf for each design
    if desInd == 3:
        DhalfDict[0] = np.array([[1,0],[1/2, np.sqrt(3)/2]])
        DhalfDict[1] = np.array([[1]])
    elif desInd == 2:
        DhalfDict[0] = np.array([[1,0],[1/2, np.sqrt(3)/2]])
    elif desInd == 1:
        DhalfDict[0] = np.array([[1]])

    # Loop through and reshape for broadcasting
    for k in np.arange(r):

        # Obtain number of random effects for this 
        # factor
        qk = nraneffs[k]

        # Reshape
        DhalfDict[k]= DhalfDict[k].reshape(*np.ones(len(dim), dtype=np.int16),qk,qk)

    # -------------------------------------------------
    # Generate smooth b maps
    # -------------------------------------------------

    # Initiate a counter
    counter = 0

    # Loop through each factor in the model
    for k in np.arange(r):

        # Obtain number of random effects for this 
        # factor
        qk = nraneffs[k]

        # Obtain number of levels for this factor
        lk = nlevels[k]

        # Obtain Dhalf for the factor
        Dhalf = DhalfDict[k]

        # Loop through each level of the factor
        for l in np.arange(lk):

            # Unsmoothed b
            b = np.random.randn(qk,*dim)

            # Reshape b for broadcasting
            b = b.transpose(1,2,3,0)
            b = b.reshape(*(b.shape),1)

            # Multiply by Dhalf (so now b has correct
            # covariance structure)
            b = Dhalf @ b

            # Reshape b back for smoothing
            b = b.reshape(b.shape[:-1])
            b = b.transpose(3,0,1,2)

            # # Smoothed b
            # b = smooth_data(b, 4, [0,fwhm,fwhm,fwhm], trunc=6, scaling='kernel')

            # Loop through each random effect and save b
            for re in np.arange(qk):

                # Save b to file
                addBlockToNifti(os.path.join(testDir,"data","b"+str(counter)+".nii"), b[re,:,:,:], np.arange(v), volInd=0,dim=dim)

                # Increment counter
                counter = counter + 1

    # -----------------------------------------------------
    # Obtain Y
    # -----------------------------------------------------

    # Work out Xbeta
    Xbeta = X @ beta

    # Loop through subjects generating nifti images
    for i in np.arange(n):    

        # Initialize Yi to Xi times beta
        Yi = Xbeta[0,i,0]

        # Loop through each random factor, factor k, adding 
        # Z_(k,j)b_k for the level j that Yi belongs to in
        # factor k 
        for k in np.arange(r):    
        
            # Get the level j of factor k which Yi belongs to
            j = fDict[k][i]

            # Get qk, number of random effects for factor k
            qk = nraneffs[k]

            # Initialize bkj volume to zeros of size (qk, *dim)
            bkj = np.zeros((*dim, qk, 1))

            # Indices for bkj
            bkjInds = get_bInds_kj(k, j, nlevels, nraneffs)

            # Loop through and read in the b volume for each
            # random effect for factor k level j.
            for re in np.arange(qk):

                # Work out which b corresponds to factor k level
                # j random effect re
                current_Ind = bkjInds[re]

                # Load in the bkj_re volume
                bkj_re = nib.load(os.path.join(testDir,"data","b"+str(current_Ind)+".nii")).get_data()

                # Add it to the bkj volume
                bkj[:,:,:,re:(re+1),0]=bkj_re

            # Obtain Z_(k) row i
            Zk_i = rrDict[k][i,:]

            # Reshape Zk_i for broadcasting
            Zk_i = Zk_i.reshape(1,1,1,1,*(Zk_i.shape))

            # Add Zk_i*bkj to Yi
            Yi = Yi + Zk_i @ bkj

        # Get epsiloni
        epsiloni = get_epsilon(v, 1).reshape(dim)

        # # Smooth epsiloni
        # epsiloni = smooth_data(epsiloni, 3, [fwhm]*3, trunc=6, scaling='kernel').reshape(dim)

        # Reshape Yi to epsiloni shape
        Yi = Yi.reshape(dim)

        # Add epsilon to Yi
        Yi = Yi + epsiloni

        # Output epsiloni
        #addBlockToNifti(os.path.join(testDir,"data","eps"+str(i)+".nii"), epsiloni, np.arange(v), volInd=0,dim=dim)

        # Output Yi unmasked
        #addBlockToNifti(os.path.join(testDir,"data","Y"+str(i)+"_unsmoothed.nii"), Yi, np.arange(v), volInd=0,dim=dim)

        # Smooth Y_i
        Yi = smooth_data(Yi, 3, [fwhm]*3, trunc=6, scaling='kernel').reshape(dim)

        # Obtain mask
        mask = get_random_mask(dim).reshape(Yi.shape)

        # Output Yi unmasked
        #addBlockToNifti(os.path.join(testDir,"data","Y"+str(i)+"_unmask.nii"), Yi, np.arange(v), volInd=0,dim=dim)

        # Save mask
        #addBlockToNifti(os.path.join(testDir,"data","M"+str(i)+".nii"), mask, np.arange(v), volInd=0,dim=dim)

        # Mask Yi
        Yi = Yi*mask

        # Truncate off (handles smoothing edge effects)
        Yi = Yi[10:(dim[0]-10),10:(dim[1]-10),10:(dim[2]-10)]

        # Output Yi
        addBlockToNifti(os.path.join(testDir,"data","Y"+str(i)+".nii"), Yi, np.arange(np.prod(origdim)), volInd=0,dim=origdim)

    # -----------------------------------------------------
    # Delete epsilons and bs
    # -----------------------------------------------------

    # Loop through bs and remove them
    for i in np.arange(q):

        # Remove b i
        os.remove(os.path.join(testDir,"data","b"+str(i)+".nii"))

    # -----------------------------------------------------
    # Save X
    # -----------------------------------------------------

    # Write out Z in full to a csv file
    pd.DataFrame(X.reshape(n,p)).to_csv(os.path.join(testDir,"data","X.csv"), header=None, index=None)

    # -----------------------------------------------------
    # Save Z
    # -----------------------------------------------------

    # Write out Z in full to a csv file
    pd.DataFrame(Z.reshape(n,q)).to_csv(os.path.join(testDir,"data","Z.csv"), header=None, index=None)

    # Write out factors in full to a csv file
    for k in np.arange(r):

        # Output raw regressors
        pd.DataFrame(rrDict[k]).to_csv(os.path.join(testDir, "data", "rr" + str(k) + ".csv"), header=None, index=None)

        # Output factor vectors
        pd.DataFrame(fDict[k]).to_csv(os.path.join(testDir, "data", "f" + str(k) + ".csv"), header=None, index=None)

    # -----------------------------------------------------
    # Inputs file
    # -----------------------------------------------------

    # Write to an inputs file
    with open(os.path.join(testDir,'inputs.yml'), 'a') as f:

        # X, Y, Z and Masks
        f.write("Y_files: " + os.path.join(testDir,"data","Yfiles.txt") + os.linesep)
        f.write("X: " + os.path.join(testDir,"data","X.csv") + os.linesep)
        f.write("Z: " + os.linesep)

        # Add each factor
        for k in np.arange(r):
            f.write("  - f" + str(k+1) + ": " + os.linesep)
            f.write("      name: f" + str(k+1) + os.linesep)
            f.write("      factor: " + os.path.join(testDir, "data", "f" + str(k) + ".csv") + os.linesep)
            f.write("      design: " + os.path.join(testDir, "data", "rr" + str(k) + ".csv") + os.linesep)

        # Output directory
        f.write("outdir: " + os.path.join(testDir,"BLMM") + os.linesep)

        # Missingness percentage
        f.write("Missingness: " + os.linesep)
        f.write("  MinPercent: " + str(rmThresh) + os.linesep)

        # Let's not output covariance maps for now!
        f.write("OutputCovB: False" + os.linesep)

        # Contrast vectors
        f.write("contrasts: " + os.linesep)
        f.write("  - c1: " + os.linesep)
        f.write("      name: null_contrast" + os.linesep)
        f.write("      vector: [0, 0, 0, 1]" + os.linesep)
        f.write("      statType: T " + os.linesep)

        # Voxel-wise batching for speedup - not necessary - but
        # convenient
        f.write("voxelBatching: 1" + os.linesep)
        f.write("MAXMEM: 2**34" + os.linesep)

        # Safe mode
        f.write("safeMode: 1" + os.linesep)

        # Log directory and test mode (backdoor options)
        f.write("test: 1" + os.linesep)
        f.write("logdir: " + os.path.join(testDir,"testlog"))

    # -----------------------------------------------------
    # Yfiles.txt
    # -----------------------------------------------------
    with open(os.path.join(testDir,"data",'Yfiles.txt'), 'a') as f:

        # Loop through listing mask files in text file
        for i in np.arange(n):

            # Write filename to text file
            if i < n-1:
                f.write(os.path.join(testDir,"data","Y"+str(i)+".nii") + os.linesep)
            else:
                f.write(os.path.join(testDir,"data","Y"+str(i)+".nii"))


    # -----------------------------------------------------
    # Version of data which can be fed into R
    # -----------------------------------------------------
    #  - i.e. seperate Y out into thousands of csv files
    #         each containing number of subjects by 1000 
    #         voxel arrays.
    # -----------------------------------------------------

    # Number of voxels in each batch
    nvb = 1000

    # Work out number of groups we have to split indices into.
    nvg = int(np.prod(origdim)//nvb)


    # Write out the number of voxel groups we split the data into
    with open(os.path.join(testDir, "data", "nb.txt"), 'w') as f:
        print(int(nvg), file=f)

    print('---------------------------------------------------------------------')
    print('Data generation complete')
    print('---------------------------------------------------------------------')

# R preprocessing
def Rpreproc(OutDir,testNo,dim,nvg,cv):

    # Get test directory
    testDir = os.path.join(OutDir, 'test' + str(testNo))

    # Make sure in numpy format
    dim = np.array(dim)

    # Number of voxels
    v = np.prod(dim)

    # There should be an inputs file in each test directory
    with open(os.path.join(testDir,'inputs.yml'), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # Number of observations
    X = pd.io.parsers.read_csv(os.path.join(testDir,"data","X.csv"), header=None).values
    n = X.shape[0]

    # Relative masking threshold
    rmThresh = inputs['Missingness']['MinPercent']

    # Split voxels we want to look at into groups we can compute
    voxelGroups = np.array_split(np.arange(v), nvg)

    # Current group of voxels
    inds_cv = voxelGroups[cv]

    # Number of voxels currently (should be ~1000)
    v_current = len(inds_cv)

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

    # Loop through each subject reading in Y and reducing to just the voxels 
    # needed
    for i in np.arange(n):

        # Load in the Y volume
        Yi = nib.load(os.path.join(testDir,"data","Y"+str(i)+".nii")).get_data()

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


        # Check if we are in safe mode (we usually will be)
        if inputs['safeMode']==1:

            # Threshold out the voxels which are underidentified (same
            # practice as lmer)
            if np.count_nonzero(Y_concat[:,vox], axis=0) <= q:

                # If we don't have enough data lets replace that voxel 
                # with zeros
                Y_concat[:,vox] = np.zeros(Y_concat[:,vox].shape)

    # Write out Z in full to a csv file
    pd.DataFrame(Y_concat.reshape(n,v_current)).to_csv(os.path.join(testDir,"data","Y_Rversion_" + str(cv) + ".csv"), header=None, index=None)

def get_random_mask(dim):

    # FWHM
    fwhm = 10

    # Load analysis mask
    mu = nib.load(os.path.join(os.path.dirname(__file__),'mask.nii')).get_data()

    # Add some noise and smooth
    mu = smooth_data(mu + 8*np.random.randn(*(mu.shape)), 3, [fwhm]*3)

    # Re-threshold (this has induced a bit of randomness in the mask shape)
    mu = 1*(mu > 0.6)

    return(mu)

def get_bInds_kj(k,j, nlevels, nraneffs):

    # Get qk
    qk = nraneffs[k]

    # Get indices representing start of
    # columns for each Z_(k)
    indices = np.cumsum(nlevels*nraneffs)
    indices = np.insert(indices,0,0)

    # Start
    start = j*qk + indices[k]

    # End
    end = (j+1)*qk + indices[k]

    # Return start and end
    return(np.arange(start,end))

def get_X(n,p):

    # Generate random X.
    X = np.random.uniform(low=-0.5,high=0.5,size=(n,p))
    
    # Make the first column an intercept
    X[:,0]=1

    # Reshape to dimensions for broadcasting
    X = X.reshape(1, n, p)

    # Return X
    return(X)

def get_Z(n, nlevels, nraneffs):

    # Work out q
    q = np.sum(nlevels*nraneffs)

    # Work out r
    r = len(nlevels)

    # Save factors and raw regressor matrices as dicts
    fDict = dict()
    rrDict = dict()

    # We need to create a block of Z for each level of each factor
    for i in np.arange(r):

        Zdata_factor = np.random.uniform(low=-0.5,high=0.5,size=(n,nraneffs[i]))

        if i==0:

            #The first factor should be block diagonal, so the factor indices are grouped
            factorVec = np.repeat(np.arange(nlevels[i]), repeats=np.floor(n/max(nlevels[i],1)))

            if len(factorVec) < n:

                # Quick fix incase rounding leaves empty columns
                factorVecTmp = np.zeros(n)
                factorVecTmp[0:len(factorVec)] = factorVec
                factorVecTmp[len(factorVec):n] = nlevels[i]-1
                factorVec = np.int64(factorVecTmp)


                # Crop the factor vector - otherwise have a few too many
                factorVec = factorVec[0:n]

                # Give the data an intercept
                Zdata_factor[:,0]=1

        else:

            # The factor is randomly arranged 
            factorVec = np.random.randint(0,nlevels[i],size=n) 

            # Ensure all levels included
            while len(np.unique(factorVec))<nlevels[i]:
                factorVec = np.random.randint(0,nlevels[i],size=n)

        # Save factor vector and raw regressor matrix
        fDict[i] = factorVec
        rrDict[i] = Zdata_factor

        # Build a matrix showing where the elements of Z should be
        indicatorMatrix_factor = np.zeros((n,nlevels[i]))
        indicatorMatrix_factor[np.arange(n),factorVec] = 1

        # Need to repeat for each random effect the factor has 
        indicatorMatrix_factor = np.repeat(indicatorMatrix_factor, nraneffs[i], axis=1)

        # Enter the Z values
        indicatorMatrix_factor[indicatorMatrix_factor==1]=Zdata_factor.reshape(Zdata_factor.shape[0]*Zdata_factor.shape[1])

        # Make sparse
        Zfactor = scipy.sparse.csr_matrix(indicatorMatrix_factor)

        # Put all the factors together
        if i == 0:
            Z = Zfactor
        else:
            Z = scipy.sparse.hstack((Z, Zfactor))

    # Convert Z to dense
    Z = Z.toarray()

    # Reshape to dimensions for broadcasting
    Z = Z.reshape(1, n, q)

    # Return Z
    return(Z, fDict, rrDict)


def get_beta(p):
    
    # Make beta (we just have beta = [p-1,p-2,...,0])
    beta = p-1-np.arange(p)

    # Reshape to dimensions for broadcasting
    beta = beta.reshape(1, p, 1)

    # Return beta
    return(beta)


def get_sigma2(v):

    # Make sigma2 (for now just set to one across all voxels)
    sigma2 = 1#np.ones(v).reshape(v,1)

    # Return sigma
    return(sigma2)

def get_epsilon(v,n):

    # Get sigma2
    sigma2 = get_sigma2(v)

    # Make epsilon.
    epsilon = sigma2*np.random.randn(v,n)

    # Reshape to dimensions for broadcasting
    epsilon = epsilon.reshape(v, n, 1)

    return(epsilon)

def get_Dhalf(v, nlevels, nraneffs):

    # Work out q
    q = np.sum(nlevels*nraneffs)

    # Empty Ddict and Dhalfdict
    Ddict = dict()
    Dhalfdict = dict()


    # Work out indices (there is one block of D per level)
    inds = np.zeros(np.sum(nlevels)+1)
    counter = 0

    for k in np.arange(len(nraneffs)):

        # Generate random D block for this factor
        Dhalfdict[k] = np.random.uniform(low=-0.5,high=0.5,size=(v,nraneffs[k],nraneffs[k]))
        Ddict[k] = Dhalfdict[k] @ Dhalfdict[k].transpose(0,2,1)

        for j in np.arange(nlevels[k]):

            # Work out indices in D corresponding to each level of the factor.
            inds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1


    # Last index will be missing so add it
    inds[len(inds)-1]=inds[len(inds)-2]+nraneffs[-1]

    # Make sure indices are ints
    inds = np.int64(inds)

    # Initial D and Dhalf
    Dhalf = np.zeros((v,q,q))
    D = np.zeros((v,q,q))

    # Fill in the blocks of D and Dhalf
    counter = 0
    for k in np.arange(len(nraneffs)):

        for j in np.arange(nlevels[k]):

            # Fill in blocks of Dhalf and D
            Dhalf[:, inds[counter]:inds[counter+1], inds[counter]:inds[counter+1]] = Dhalfdict[k]
            D[:, inds[counter]:inds[counter+1], inds[counter]:inds[counter+1]] = Ddict[k]

            # Increment counter
            counter=counter+1

    return(Dhalfdict)

def get_b(v, nlevels, nraneffs):

    # Work out q
    q = np.sum(nlevels*nraneffs)
    
    # Make random b
    b = np.random.randn(v*q).reshape(v,q,1)

    # Get cholesky root of D
    Dhalf = get_Dhalf(v, nlevels, nraneffs)

    # Give b the correct covariance structure
    b = Dhalf @ b

    # Reshape to dimensions for broadcasting
    b = b.reshape(v, q, 1)

    # Return b
    return(b)


def get_Y(X, Z, beta, b, epsilon):

    # Generate the response vector
    Y = X @ beta + Z @ b + epsilon

    # Return Y
    return(Y)



# ============================================================================
#
# The below function adds a block of voxels to a pre-existing NIFTI or creates
# a NIFTI of specified dimensions if not.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `fname`: An absolute path to the Nifti file.
# - `block`: The block of values to write to the NIFTI.
# - `blockInds`: The indices representing the 3D coordinates `block` should be 
#                written to in the NIFTI. (Note: It is assumed if the NIFTI is
#                4D we assume that the indices we want to write to in each 3D
#                volume/slice are the same across all 3D volumes/slices).
# - `dim` (optional): If creating the NIFTI image for the first time, the 
#                     dimensions of the NIFTI image must be specified.
# - `volInd` (optional): If we only want to write to one 3D volume/slice,
#                        within a 4D file, this specifies the index of the
#                        volume of interest.
# - `aff` (optional): If creating the NIFTI image for the first time, the 
#                     affine of the NIFTI image must be specified.
# - `hdr` (optional): If creating the NIFTI image for the first time, the 
#                     header of the NIFTI image must be specified.
#
# ============================================================================
def addBlockToNifti(fname, block, blockInds,dim=None,volInd=None,aff=None,hdr=None):

    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            f = os.open(fname + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True

    # Check volInd is correct datatype
    if volInd is not None:

        volInd = int(volInd)

    # Check whether the NIFTI exists already
    if os.path.isfile(fname):

        # Work out dim if we don't already have it
        dim = nib.Nifti1Image.from_filename(fname, mmap=False).shape

        # Work out data
        data = nib.Nifti1Image.from_filename(fname, mmap=False).get_fdata().copy()

        # Work out affine
        affine = nib.Nifti1Image.from_filename(fname, mmap=False).affine.copy()
        
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
    if volInd is None:

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
        data = data[:,volInd].reshape((n_vox,1))

        # Add block
        data[blockInds,:] = block.reshape(data[blockInds,:].shape)
        
        # Put in the volume
        data_out[:,:,:,volInd] = data[:,0].reshape(int(dim[0]),
                                                 int(dim[1]),
                                                 int(dim[2]))
    
    # Save NIFTI
    nib.save(nib.Nifti1Image(data_out, affine, header=hdr), fname)

    # Delete lock file, so other jobs know they can now write to the
    # file
    os.remove(fname + ".lock")
    os.close(f)

    del fname, data_out, affine, data, dim


# Smoothing function
def smooth_data(data, D, fwhm, trunc=6, scaling='kernel'):

    # -----------------------------------------------------------------------
    # Reformat fwhm
    # -----------------------------------------------------------------------

    # Format fwhm and replace None with 0
    fwhm = np.asarray([fwhm]).ravel()
    fwhm = np.asarray([0. if elem is None else elem for elem in fwhm])

    # Non-zero dimensions
    D_nz = np.sum(fwhm>0)

    # Convert fwhm to sigma values
    sigma = fwhm / np.sqrt(8 * np.log(2))

    # -----------------------------------------------------------------------
    # Perform smoothing (this code is based on `_smooth_array` from the
    # nilearn package)
    # -----------------------------------------------------------------------
    
    # Loop through each dimension and smooth
    for n, s in enumerate(sigma):

        # If s is non-zero smooth by s in that direction.
        if s > 0.0:

            # Perform smoothing in nth dimension
            ndimage.gaussian_filter1d(data, s, output=data, mode='constant', axis=n, truncate=trunc)


    # -----------------------------------------------------------------------
    # Rescale
    # -----------------------------------------------------------------------
    if scaling=='kernel':
    
        # -----------------------------------------------------------------------
        # Rescale smoothed data to standard deviation 1 (this code is based on
        # `_gaussian_kernel1d` from the `scipy.ndimage` package).
        # -----------------------------------------------------------------------

        # Calculate sigma^2
        sigma2 = sigma*sigma

        # Calculate kernel radii
        radii = np.int16(trunc*sigma + 0.5)

        # Initialize array for phi values (has to be object as dimensions can 
        # vary in length)
        phis = np.empty(shape=(D_nz),dtype=object)

        # Index for non-zero dimensions
        j = 0

        # Loop through dimensions to get scaling constants
        for k in np.arange(D):

            # Skip the non-smoothed dimensions
            if fwhm[k]!=0:

                # Get range of values for this dimension
                r = np.arange(-radii[k], radii[k]+1)
                
                # Get the kernel for this dimension
                phi = np.exp(-0.5 / sigma2[k] * r ** 2)

                # Normalise phi
                phi = phi / phi.sum()

                # Add phi to dictionary
                phis[j]= phi[::-1]

                # Increment j
                j = j + 1
                
        # Create the D_nz dimensional grid
        grids = np.meshgrid(*phis);

        # Initialize normalizing constant
        ss = 1

        # Loop through axes and take products
        for j in np.arange(D_nz-1):

            # Smoothing kernel along plane (j,j+1)
            product_gridj = (grids[j]*(grids[j+1]*np.ones(grids[0].shape)).T)

            # Get normalizing constant along this plane
            ssj = np.sum((product_gridj)**2)

            # Add to running smoothing constant the sum of squares of this kernel
            # (Developer note: This is the normalizing constant. When you smooth
            # you are mutliplying everything by a grid of values along each dimension.
            # To restandardize you then need to take the sum the squares of this grid
            # and squareroot it. You then divide your data by this number at the end.
            # This must be done once for every dimension, hence the below product.)
            ss = ssj*ss

        # Rescale noise
        data = data/np.sqrt(ss)

    elif scaling=='max':

        # Rescale noise by dividing by maximum value
        data = data/np.max(data)

    return(data)

# #generate_data(n,dim,OutDir,testNo,desInd)
#generate_data(10, np.array([100,100,100]), '/home/tommaullin/Documents/BLMM/test/', 23, 2)


# nvb = 1000

# # Work out number of groups we have to split indices into.
# nvg = int(100**3//nvb)
# #Rpreproc('$1', $2, [100,100,100], $3, $4)
# for i in np.arange(400,600):
#     Rpreproc('/home/tommaullin/Documents/BLMM/test/',20,[100,100,100],nvg,i)