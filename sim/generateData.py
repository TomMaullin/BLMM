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
def get_data(n,dim,fwhm,OutDir):

    # Make new data directory.
    if not os.path.exists(os.path.join(OutDir,"data")):
        os.mkdir(os.path.join(OutDir,"data"))

    # -------------------------------------------------
    # Design parameters
    # -------------------------------------------------

    # Number of fixed effects parameters
    p = 4

    # Number of random effects grouping factors
    r = 2

    # Number of levels for each factor
    nlevels = np.array([20,10])

    # Number of levels for each factor
    nraneffs = np.array([2,1])

    # Number of voxels
    v = np.prod(dim)

    # Second dimension of Z
    q = np.sum(nlevels*nraneffs)


    print('n: ', n, ', q: ', q, ', p: ', p, ', v:', v)

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
    # Generate smooth epsilon maps
    # -------------------------------------------------

    # Generate a smooth epsilon map for each subject
    for s in np.arange(n):

        # Get epsilon
        epsilon = get_epsilon(v, 1).reshape(dim)

        # Smooth epsilon
        epsilon = smooth_data(epsilon, 3, [fwhm]*3, trunc=6, scaling='kernel')

        # Save epsilon to file
        addBlockToNifti(os.path.join(OutDir,"data","epsilon"+str(s)+".nii"), epsilon, np.arange(v), volInd=0,dim=dim)


    # -------------------------------------------------
    # Generate smooth b maps
    # -------------------------------------------------

    # Initiate a counter
    counter = 0

    # Obtain the dictionary of cholesky factors of 
    # diagonal blocks of D
    #Dhalfdict = get_Dhalf(v, nlevels, nraneffs)

    # Loop through each factor in the model
    for k in np.arange(r):

        # Obtain number of random effects for this 
        # factor
        qk = nraneffs[k]

        # Obtain number of levels for this factor
        lk = nlevels[k]

        # Obtain Dhalf for the factor
        Dhalf = np.random.randn(*dim,qk,qk)#Dhalfdict[k].reshape(*(dim),qk,qk)

        # Loop through each level of the factor
        for l in np.arange(lk):

            # Unsmoothed b
            b = np.random.randn(qk,*dim)

            # Reshape b for broadcasting
            b = b.transpose(1,2,3,0)
            b = b.reshape(*(b.shape),1)

            print(Dhalf.shape)
            print(b.shape)

            # Multiply by Dhalf (so now b has correct
            # covariance structure)
            b = Dhalf @ b

            # Reshape b back for smoothing
            b = b.reshape(b.shape[:-1])
            b = b.transpose(3,0,1,2)

            # Smoothed b
            b = smooth_data(b, 4, [0,fwhm,fwhm,fwhm], trunc=6, scaling='kernel')

            # Loop through each random effect and save b
            for q in np.arange(qk):

                # Save b to file
                addBlockToNifti(os.path.join(OutDir,"data","b"+str(counter)+".nii"), b[q,:,:,:], np.arange(v), volInd=0,dim=dim)

                # Increment counter
                counter = counter + 1

    # -----------------------------------------------------
    # Obtain Y
    # -----------------------------------------------------

    print('X shape ', X.shape)
    print('nraneffs ', nraneffs)

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
                bkj_re = nib.load(os.path.join(OutDir,"data","b"+str(current_Ind)+".nii")).get_data()

                # Add it to the bkj volume
                bkj[:,:,:,re:(re+1),0]=bkj_re

            # Obtain Z_(k) row i
            Zk_i = rrDict[k][i,:]

            # Reshape Zk_i for broadcasting
            Zk_i = Zk_i.reshape(1,1,1,1,*(Zk_i.shape))

            # Add Zk_i*bkj to Yi
            Yi = Yi + Zk_i @ bkj

        # Load in epsilon_i
        epsiloni = nib.load(os.path.join(OutDir,"data","epsilon"+str(i)+".nii")).get_data()

        # Reshape Yi to epsiloni shape
        Yi = Yi.reshape(epsiloni.shape)

        # Add epsilon to Yi
        Yi = Yi + epsiloni

        # Output Yi
        addBlockToNifti(os.path.join(OutDir,"data","Y"+str(i)+".nii"), Yi, np.arange(v), volInd=0,dim=dim)

    
    # # -------------------------------------------------
    # # Get spherical mask
    # # -------------------------------------------------

    #    # Obtain non-random mask
    #    mask = get_sphere(dim)

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
    X = np.random.randn(n,p)
    
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

        Zdata_factor = np.random.randn(n,nraneffs[i])

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
        Dhalfdict[k] = np.random.randn(v,nraneffs[k],nraneffs[k])
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

    print(fname)
    print(os.path.isfile(fname))

    # Check whether the NIFTI exists already
    if os.path.isfile(fname):

        print('here')

        # Load in NIFTI
        #img = nib.Nifti1Image.from_filename(fname, mmap=False)

        # Work out dim if we don't already have it
        dim = nib.Nifti1Image.from_filename(fname, mmap=False).shape

        # print('in mem 1 ', img.in_memory)

        # Work out data
        data = nib.Nifti1Image.from_filename(fname, mmap=False).get_fdata().copy()
        # data_array = np.asarray(img.dataobj)

        # print('in mem 2 ', img.in_memory)

        # Work out affine
        affine = nib.Nifti1Image.from_filename(fname, mmap=False).affine.copy()

        # # Delete image
        # img.uncache()

        # print('in mem 3 ', img.in_memory)
        # del img
        
    else:

        print('here2')

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


    # Make NIFTI
    #nifti = nib.Nifti1Image(data_out, affine, header=hdr)
    
    #print('in mem 4 ', nifti.in_memory)
    
    # Save NIFTI
    nib.save(nib.Nifti1Image(data_out, affine, header=hdr), fname)

    # Uncache NIFTI
    #nifti.uncache()

    #print('in mem 5 ', nifti.in_memory)

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

    print('fwhm  ', fwhm)

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

        # Initialize empty product grid
        product_grid = np.ones(grids[0].shape)

        # Loop through axes and take products
        for j in np.arange(D_nz):

            product_grid = grids[j]*product_grid

        # Get the normalizing constant by summing over grid
        ss = np.sum(product_grid**2)

        # Rescale noise
        data = data/np.sqrt(ss)

    elif scaling=='max':

        # Rescale noise by dividing by maximum value
        data = data/np.max(data)

    return(data)


get_data(100, np.array([100,100,100]), 5, '/home/tommaullin/Documents/BLMM/sim/')