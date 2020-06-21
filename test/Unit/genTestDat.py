import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
import os
import shutil

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.npMatrix2d import vech2mat2D

# =============================================================================
# This file contains functions for generating testdata and product matrices,
# used by the BLMM unit tests.
#
# Author: Tom Maullin
# Last edited: 06/04/2020
#
# =============================================================================

# =============================================================================
#
# The below function generates a random testcase according to the linear mixed
# model:
#
#   Y = X\beta + Zb + \epsilon
#
# Where b~N(0,D) and \epsilon ~ N(0,\sigma^2 I)
#
# -----------------------------------------------------------------------------
#
# It takes the following inputs:
#
# -----------------------------------------------------------------------------
#
#   - n (optional): Number of observations. If not provided, a random n will be
#                   selected between 800 and 1200.
#   - p (optional): Number of fixed effects parameters. If not provided, a
#                   random p will be selected between 2 and 10 (an intercept is
#                   automatically included).
#   - nlevels (optional): A vector containing the number of levels for each
#                         random factor, e.g. `nlevels=[3,4]` would mean the
#                         first factor has 3 levels and the second factor has
#                         4 levels. If not provided, default values will be
#                         between 8 and 40.
#   - nraneffs (optional): A vector containing the number of random effects for
#                          each factor, e.g. `nraneffs=[2,1]` would mean the 
#                          first factor has random effects and the second
#                          factor has 1 random effect.
#   - save (optional): Boolean value. If true, the design will be saved.
#   - desInd (optional): Integer value. Index representing which design is
#                        being run, only needed for saving.
#   - simInd (optional): Integer value. Index representing which simulation is
#                        being run, only needed for saving.
#   - OutDir (optional): Output directory to save the design to, if saving.
#
# -----------------------------------------------------------------------------
#
# And gives the following outputs:
#
# -----------------------------------------------------------------------------
#
#   - X: A fixed effects design matrix of dimensions (n x p) including a random
#        intercept column (the first column).
#   - Y: A response vector of dimension (n x 1).
#   - Z: A random effects design matrix of size (n x q) where q is equal to the
#        product of nlevels and nraneffs.
#   - nlevels: A vector containing the number of levels for each random factor,
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#   - beta: The true values of beta used to simulate the response vector.
#   - sigma2: The true value of sigma2 used to simulate the response vector.
#   - D: The random covariance matrix used to simulate b and the response vector.
#   - b: The random effects vector used to simulate the response vector.
#
# -----------------------------------------------------------------------------
def genTestData2D(n=None, p=None, nlevels=None, nraneffs=None, save=False, simInd=None, desInd=0, OutDir=None, factorVectors=None, X=None, Z=None):

    # Check if we have n
    if n is None:

        # If not generate a random n
        n = np.random.randint(800,1200)
    
    # Check if we have p
    if p is None:

        # If not generate a random p
        p = np.random.randint(2,10)

    # Work out number of random factors.
    if nlevels is None and nraneffs is None:

        # If we have neither nlevels or nraneffs, decide on a number of
        # random factors, r.
        r = np.random.randint(2,4)

    elif nlevels is None:

        # Work out number of random factors, r
        r = np.shape(nraneffs)[0]

    else:

        # Work out number of random factors, r
        r = np.shape(nlevels)[0]

    # Check if we need to generate nlevels.
    if nlevels is None:
        
        # Generate random number of levels.
        nlevels = np.random.randint(8,40,r)

    # Check if we need to generate nraneffs.
    if nraneffs is None:
        
        # Generate random number of levels.
        nraneffs = np.random.randint(2,5,r)

    # Assign outdir and simInd if needed:
    if save:
        if not OutDir:
            OutDir = os.getcwd()
        if not simInd:
            simInd = ''
        if not desInd:
            desInd = ''

    # If a design matrix has not been specified make one.
    if X is None:

        # Generate random X.
        X = np.random.randn(n,p)
        
        # Make the first column an intercept
        X[:,0]=1

    # Dictionary of factor vectors
    if factorVectors is None:
        factorVectors = dict()

    if Z is None:

        # Create Z
        # We need to create a block of Z for each level of each factor
        for i in np.arange(r):

            Zdata_factor = np.random.randn(n,nraneffs[i])

            if i==0:

                if 'i' not in factorVectors:

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

                    # If we specified a factor vec to use, use it
                    factorVec = factorVectors[i]

                    # Give the data an intercept
                    Zdata_factor[:,0]=1

            else:

                if 'i' not in factorVectors: 
        
                    # The factor is randomly arranged 
                    factorVec = np.random.randint(0,nlevels[i],size=n) 
                    while len(np.unique(factorVec))<nlevels[i]:
                        factorVec = np.random.randint(0,nlevels[i],size=n)

                else:

                    # The factor is randomly arranged 
                    factorVec = factorVectors[i]

            # Save the factor vectors if we don't have it already
            if 'i' not in factorVectors:

                factorVectors[i]=factorVec

            # If outputting, save the factor vector
            if save:
                np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zfactor' + str(i) + '.csv'), factorVec)
                np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zdata' + str(i) + '.csv'), Zdata_factor)


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

    else:

        # If outputting, save the factor vector
        if save:
            for i in range(len(nraneffs)):
                shutil.copyfile(os.path.join(OutDir, 'Sim1_Design' + str(desInd) + '_Zfactor' + str(i) + '.csv'),os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zfactor' + str(i) + '.csv'))
                shutil.copyfile(os.path.join(OutDir, 'Sim1_Design' + str(desInd) + '_Zdata' + str(i) + '.csv'),os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zdata' + str(i) + '.csv'))

    # Make random beta
    if desInd==0:
        beta = np.random.randint(-5,5,p).reshape(p,1)
    else:
        beta = p-np.arange(1,p+1).reshape(p,1)


    if desInd==0:
        # Make random sigma2
        sigma2 = 0.5*np.random.randn()**2
    else:
        sigma2 = 1

    # Make epsilon.
    epsilon = sigma2*np.random.randn(n).reshape(n,1)

    # Make random D
    Ddict = dict()
    Dhalfdict = dict()
    if desInd==0:
    
        for k in np.arange(r):
      
            # Create a random matrix
            randMat = np.random.uniform(-1,1,(nraneffs[k],nraneffs[k]))

            # Record it as D^{1/2}
            Dhalfdict[k] = randMat

            # Work out D = D^{1/2} @ D^{1/2}'
            Ddict[k] = randMat @ randMat.transpose()

    elif desInd==1 or desInd==4:

        Ddict[0]=np.eye(2)

    elif desInd==2:

        Ddict[0]= vech2mat2D(np.array([[1,0.7,0.5,0.9,0.6,0.8]]).transpose())
        Ddict[1]= np.eye(2)

    elif desInd==3:

        Ddict[0]=vech2mat2D(np.array([[1,0.75,0.5,0.25,1,0.75,0.5,1,0.75,1]]).transpose())
        Ddict[1]=vech2mat2D(np.array([[1,0.7,0.5,0.9,0.6,0.8]]).transpose())
        Ddict[2]=np.eye(2)

    if desInd!=0:
    
        for k in np.arange(r):

            # Record D^{1/2}
            Dhalfdict[k] = np.linalg.cholesky(Ddict[k])


    # Matrix version
    D = np.array([])
    Dhalf = np.array([])
    for i in np.arange(r):
      
        for j in np.arange(nlevels[i]):
        
            if i == 0 and j == 0:

                D = Ddict[i]
                Dhalf = Dhalfdict[i]

            else:

                D = scipy.linalg.block_diag(D, Ddict[i])
                Dhalf = scipy.linalg.block_diag(Dhalf, Dhalfdict[i])


    # Make random b
    q = np.sum(nlevels*nraneffs)
    b = np.random.randn(q).reshape(q,1)

    # Give b the correct covariance structure
    b = Dhalf @ b

    # Generate the response vector
    Y = X @ beta + Z @ b + epsilon

    if save:
        np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_X' + '.csv'), X)
        np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Y' + '.csv'), Y)

    # Return values
    return(Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D, factorVectors)


# =============================================================================
#
# The below function generates the product matrices from matrices X, Y and Z.
#
# -----------------------------------------------------------------------------
#
# It takes as inputs:
#
# -----------------------------------------------------------------------------
#
#  - `X`: The design matrix of dimension n times p.
#  - `Y`: The response vector of dimension n times 1.
#  - `Z`: The random effects design matrix of dimension n times q.
#
# -----------------------------------------------------------------------------
#
# It returns as outputs:
#
# -----------------------------------------------------------------------------
#
#  - `XtX`: X transposed multiplied by X.
#  - `XtY`: X transposed multiplied by Y.
#  - `XtZ`: X transposed multiplied by Z.
#  - `YtX`: Y transposed multiplied by X.
#  - `YtY`: Y transposed multiplied by Y.
#  - `YtZ`: Y transposed multiplied by Z.
#  - `ZtX`: Z transposed multiplied by X.
#  - `ZtY`: Z transposed multiplied by Y.
#  - `ZtZ`: Z transposed multiplied by Z.
#
# =============================================================================
def prodMats2D(Y,Z,X):

    # Work out the product matrices
    XtX = X.transpose() @ X
    XtY = X.transpose() @ Y
    XtZ = X.transpose() @ Z
    YtX = XtY.transpose()
    YtY = Y.transpose() @ Y
    YtZ = Y.transpose() @ Z
    ZtX = XtZ.transpose()
    ZtY = YtZ.transpose()
    ZtZ = Z.transpose() @ Z

    # Return product matrices
    return(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ)


# =============================================================================
#
# The below function generates a random testcase according to the mass 
# univariate linear mixed model:
#
#   Y_v = X_v\beta_v + Z_vb_v + \epsilon_v
#
# Where b_v~N(0,D_v) and \epsilon_v ~ N(0,\sigma_v^2 I) and the subscript 
# v reprsents that we have one such model for every voxel.
#
# Note: By default this function will always generate multi-factored examples.
#       This is to stress test the code. If you wish to generate one factor
#       data you have to specify it manually using the below options.
#
# -----------------------------------------------------------------------------
#
# It takes the following inputs:
#
# -----------------------------------------------------------------------------
#
#   - n (optional): Number of observations. If not provided, a random n will be
#                   selected between 800 and 1200.
#   - p (optional): Number of fixed effects parameters. If not provided, a
#                   random p will be selected between 2 and 10 (an intercept is
#                   automatically included).
#   - nlevels (optional): A vector containing the number of levels for each
#                         random factor, e.g. `nlevels=[3,4]` would mean the
#                         first factor has 3 levels and the second factor has
#                         4 levels. If not provided, default values will be
#                         between 8 and 40.
#   - nraneffs (optional): A vector containing the number of random effects for
#                          each factor, e.g. `nraneffs=[2,1]` would mean the
#                          first factor has random effects and the second 
#                          factor has 1 random effect.
#   - v (optional): Number of voxels. If not provided a random v will be 
#                   selected between 100 and 250.
#
# -----------------------------------------------------------------------------
#
# And gives the following outputs:
#
# -----------------------------------------------------------------------------
#
#   - X: A fixed effects design matrix of dimensions (n x p) including a random
#        intercept column (the first column).
#   - Y: A response vector of dimension (n x 1).
#   - Z: A random effects design matrix of size (n x q) where q is equal to the
#        product of nlevels and nraneffs.
#   - nlevels: A vector containing the number of levels for each random factor,
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
#   - `nraneffs`: A vector containing the number of random effects for each
#                 factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#                 random effects and the second factor has 1 random effect.
#   - beta: The true values of beta used to simulate the response vector.
#   - sigma2: The true value of sigma2 used to simulate the response vector.
#   - D: The random covariance matrix used to simulate b and the response vector.
#   - b: The random effects vector used to simulate the response vector.
#   - X_sv: A spatially varying design (X with random rows removed across 
#           voxels).
#   - Z_sv: A spatially varying random effects design (Z with random rows
#           removed across voxels).
#   - n_sv: Spatially varying number of observations.
#
# -----------------------------------------------------------------------------
def genTestData3D(n=None, p=None, nlevels=None, nraneffs=None, v=None):

    # Check if we have n
    if n is None:

        # If not generate a random n
        n = np.random.randint(800,1200)
    
    # Check if we have v
    if v is None:

        # If not generate a random n
        v = np.random.randint(100,250)

    # Check if we have p
    if p is None:

        # If not generate a random p
        p = np.random.randint(2,10)

    # Work out number of random factors.
    if nlevels is None and nraneffs is None:

        # If we have neither nlevels or nraneffs, decide on a number of
        # random factors, r.
        r = np.random.randint(2,4)

    elif nlevels is None:

        # Work out number of random factors, r
        r = np.shape(nraneffs)[0]

    else:

        # Work out number of random factors, r
        r = np.shape(nlevels)[0]

    # Check if we need to generate nlevels.
    if nlevels is None:
        
        # Generate random number of levels.
        nlevels = np.random.randint(8,40,r)

    # Check if we need to generate nraneffs.
    if nraneffs is None:
        
        # Generate random number of levels.
        nraneffs = np.random.randint(2,5,r)

    # Generate random X.
    X = np.random.randn(n,p)

    # Work out q
    q = np.sum(nlevels*nraneffs)
    
    # Make the first column an intercept
    X[:,0]=1

    # Create Z
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

    # Make random beta
    beta = np.random.randint(-5,5,v*p).reshape(v,p,1)

    # Make random sigma2
    sigma2 = (0.5*np.random.randn(v)**2).reshape(v,1,1)

    # Make epsilon.
    epsilon = sigma2*np.random.randn(n).reshape(n,1)

    # Reshape sigma2
    sigma2 = sigma2.reshape(v)

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

    # Make random b
    b = np.random.randn(v*q).reshape(v,q,1)

    # Give b the correct covariance structure
    b = Dhalf @ b

    # Generate a mask based on voxels
    mask = np.random.binomial(1,0.9,size=(v,n,1))

    # Spatially varying n
    n_sv = np.sum(mask,axis=1)

    # Work out spatially varying Z and X
    X_sv = mask*X
    Z_sv = mask*Z

    # Generate the response vector
    Y = X_sv @ beta + Z_sv @ b + epsilon

    # Return values
    return(Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv)


# =============================================================================
#
# The below function generates the product matrices from matrices X, Y and Z.
#
# -----------------------------------------------------------------------------
#
# It takes as inputs:
#
# -----------------------------------------------------------------------------
#
#  - `X`: The design matrix of dimension n times p.
#  - `Y`: The response vector of dimension v times n times 1*.
#  - `Z`: The random effects design matrix of dimension n times q.
#  - `X_sv`: The spatially varying design matrix of dimension v times n times 
#            p.
#  - `Z_sv`: The spatially varying random effects design matrix of dimension v 
#         times n times q.
#
#  *Note: Y is always assumed to vary across voxels so we do not bother with the
#         `_sv` subscript for Y.
#
# -----------------------------------------------------------------------------
#
# It returns as outputs:
#
# -----------------------------------------------------------------------------
#
#  - `XtX`: X transposed multiplied by X.
#  - `XtY`: X transposed multiplied by Y.
#  - `XtZ`: X transposed multiplied by Z.
#  - `YtX`: Y transposed multiplied by X.
#  - `YtY`: Y transposed multiplied by Y.
#  - `YtZ`: Y transposed multiplied by Z.
#  - `ZtX`: Z transposed multiplied by X.
#  - `ZtY`: Z transposed multiplied by Y.
#  - `ZtZ`: Z transposed multiplied by Z.
#  - `XtX_sv`: X_sv transposed multiplied by X_sv.
#  - `XtY_sv`: X_sv transposed multiplied by Y.
#  - `XtZ_sv`: X_sv transposed multiplied by Z_sv.
#  - `YtX_sv`: Y transposed multiplied by X_sv.
#  - `YtZ_sv`: Y transposed multiplied by Z_sv.
#  - `ZtX_sv`: Z_sv transposed multiplied by X_sv.
#  - `ZtY_sv`: Z_sv transposed multiplied by Y.
#  - `ZtZ_sv`: Z_sv transposed multiplied by Z_sv.
#
# =============================================================================
def prodMats3D(Y,Z,X,Z_sv,X_sv):

    # Work out the product matrices (non spatially varying)
    XtX = (X.transpose() @ X).reshape(1, X.shape[1], X.shape[1])
    XtY = X.transpose() @ Y
    XtZ = (X.transpose() @ Z).reshape(1, X.shape[1], Z.shape[1])
    YtX = XtY.transpose(0,2,1)
    YtY = Y.transpose(0,2,1) @ Y
    YtZ = Y.transpose(0,2,1) @ Z
    ZtX = XtZ.transpose(0,2,1)
    ZtY = YtZ.transpose(0,2,1)
    ZtZ = (Z.transpose() @ Z).reshape(1, Z.shape[1], Z.shape[1])

    # Spatially varying product matrices
    XtX_sv = X_sv.transpose(0,2,1) @ X_sv
    XtY_sv = X_sv.transpose(0,2,1) @ Y
    XtZ_sv = X_sv.transpose(0,2,1) @ Z_sv
    YtX_sv = XtY_sv.transpose(0,2,1)
    YtZ_sv = Y.transpose(0,2,1) @ Z_sv
    ZtX_sv = XtZ_sv.transpose(0,2,1)
    ZtY_sv = YtZ_sv.transpose(0,2,1)
    ZtZ_sv = Z_sv.transpose(0,2,1) @ Z_sv

    # Return product matrices
    return(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv)

