import os
import sys
import numpy as np
import cvxopt
import pandas as pd
import os
import time
import scipy.sparse
import scipy.sparse.linalg
import sys
import nibabel as nib
import nilearn

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.npMatrix2d import *
from lib.npMatrix3d import *
# =============================================================================
# This file contains all unit tests for the functions given in the tools3D.py
# file.
#
# Author: Tom Maullin
# Last edited: 22/03/2020
#
# =============================================================================

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
# -----------------------------------------------------------------------------
#
# It takes the following inputs:
#
# -----------------------------------------------------------------------------
#
#   - n (optional): Number of subjects. If not provided, a random n will be
#                   selected between 800 and 1200.
#   - p (optional): Number of fixed effects parameters. If not provided, a
#                   random p will be selected between 2 and 10 (an intercept is
#                   automatically included).
#   - nlevels (optional): A vector containing the number of levels for each
#                         random factor, e.g. `nlevels=[3,4]` would mean the
#                         first factor has 3 levels and the second factor has
#                         4 levels. If not provided, default values will be
#                         between 8 and 40.
#   - nparams (optional): A vector containing the number of parameters for each
#                         factor, e.g. `nlevels=[2,1]` would mean the first
#                         factor has 2 parameters and the second factor has 1
#                         parameter. If not provided, default values will be
#                         between 2 and 5.
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
#        product of nlevels and nparams.
#   - nlevels: A vector containing the number of levels for each random factor,
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
#   - nparams: A vector containing the number of parameters for each factor,
#              e.g. `nlevels=[2,1]` would mean the first factor has 2
#              parameters and the second factor has 1 parameter.
#   - beta: The true values of beta used to simulate the response vector.
#   - sigma2: The true value of sigma2 used to simulate the response vector.
#   - D: The random covariance matrix used to simulate b and the response vector.
#   - b: The random effects vector used to simulate the response vector.
#   - X_sv: A spatially varying design (X with random rows removed across 
#           voxels).
#   - Z_sv: A spatially varying random effects design (Z with random rows
#           removed across voxels).
#   - n_sv: Spatially varying number of subjects.
#
# -----------------------------------------------------------------------------
def genTestData(n=None, p=None, nlevels=None, nparams=None, v=None):

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
    if nlevels is None and nparams is None:

        # If we have neither nlevels or nparams, decide on a number of
        # random factors, r.
        r = np.random.randint(2,4)

    elif nlevels is None:

        # Work out number of random factors, r
        r = np.shape(nparams)[0]

    else:

        # Work out number of random factors, r
        r = np.shape(nlevels)[0]

    # Check if we need to generate nlevels.
    if nlevels is None:
        
        # Generate random number of levels.
        nlevels = np.random.randint(8,40,r)

    # Check if we need to generate nparams.
    if nparams is None:
        
        # Generate random number of levels.
        nparams = np.random.randint(2,5,r)

    # Generate random X.
    X = np.random.randn(n,p)

    # Work out q
    q = np.sum(nlevels*nparams)
    
    # Make the first column an intercept
    X[:,0]=1

    # Create Z
    # We need to create a block of Z for each level of each factor
    for i in np.arange(r):

        Zdata_factor = np.random.randn(n,nparams[i])

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

            # The factor is randomly arranged across subjects
            factorVec = np.random.randint(0,nlevels[i],size=n) 

        # Build a matrix showing where the elements of Z should be
        indicatorMatrix_factor = np.zeros((n,nlevels[i]))
        indicatorMatrix_factor[np.arange(n),factorVec] = 1

        # Need to repeat for each parameter the factor has 
        indicatorMatrix_factor = np.repeat(indicatorMatrix_factor, nparams[i], axis=1)

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

    for k in np.arange(len(nparams)):

        # Generate random D block for this factor
        Dhalfdict[k] = np.random.randn(v,nparams[k],nparams[k])
        Ddict[k] = Dhalfdict[k] @ Dhalfdict[k].transpose(0,2,1)

        for j in np.arange(nlevels[k]):

            # Work out indices in D corresponding to each level of the factor.
            inds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nparams)))[k] + nparams[k]*j
            counter = counter + 1


    # Last index will be missing so add it
    inds[len(inds)-1]=inds[len(inds)-2]+nparams[-1]

    # Make sure indices are ints
    inds = np.int64(inds)

    # Initial D and Dhalf
    Dhalf = np.zeros((v,q,q))
    D = np.zeros((v,q,q))

    # Fill in the blocks of D and Dhalf
    counter = 0
    for k in np.arange(len(nparams)):

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
    return(Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv)


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
def prodMats(Y,Z,X,Z_sv,X_sv):

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


# =============================================================================
#
# The below function tests the function `kron3D`. It does this by generating
# random block matrix input and checking against numpy.
#
# =============================================================================
def test_kron3D():

    # Random dimensions (v,a1,b1) and (v,a2,b2)
    v = np.random.randint(100,250)
    a1 = np.random.randint(5,10)
    a2 = np.random.randint(5,10)
    b1 = np.random.randint(5,10)
    b2 = np.random.randint(5,10)

    # Random matrices of said dimensions (corresponding to spatially varying
    # case)
    A_sv = np.random.randn(v,a1,b1)
    B_sv = np.random.randn(v,a2,b2)

    # Now make A and B corresponding to non-spatially varying case.
    A = np.random.randn(1,a1,b1)
    B = np.random.randn(1,a2,b2)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test spatially varying
    result_sv = np.allclose(kron3D(A_sv,B_sv)[testv,:,:], np.kron(A_sv[testv,:,:],B_sv[testv,:,:]))

    # Test non spatially varying
    result1_nsv = np.allclose(kron3D(A_sv,B)[testv,:,:], np.kron(A_sv[testv,:,:],B[0,:,:]))
    result2_nsv = np.allclose(kron3D(A,B_sv)[testv,:,:], np.kron(A[0,:,:],B_sv[testv,:,:]))

    # Combine results
    testVal = result_sv and result1_nsv and result2_nsv

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: kron3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `mat2vec3D`. It does this by generating
# a random example and testing against it's 2D counterpart from tools2d.py.
#
# =============================================================================
def test_mat2vec3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,100)
    a = np.random.randint(10,20)
    b = np.random.randint(10,20)

    # Generate random matrix
    A = np.random.randn(v,a,b)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D version.
    testVal = np.allclose(mat2vec3D(A)[testv,:,:], mat2vec2D(A[testv,:,:]))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mat2vec3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `mat2vech3D`. It does this by 
# generating a random example and testing against it's 2D counterpart from 
# tools2d.py.
#
# =============================================================================
def test_mat2vech3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,100)
    a = np.random.randint(10,20)

    # Generate random matrix
    A = np.random.randn(v,a,a)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D version.
    testVal = np.allclose(mat2vech3D(A)[testv,:,:], mat2vech2D(A[testv,:,:]))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mat2vech3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `vec2mat3D`. It does this by 
# generating a random example and testing against it's 2D counterpart from 
# tools2d.py.
#
# =============================================================================
def test_vec2mat3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,100)
    a = np.random.randint(10,20)

    # Generate random matrix
    A = np.random.randn(v,a**2,1)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D version.
    testVal = np.allclose(vec2mat3D(A)[testv,:,:], vec2mat2D(A[testv,:,:]))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: vec2mat3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `vech2mat3D`. It does this by 
# generating a random example and testing against it's 2D counterpart from 
# tools2d.py.
#
# =============================================================================
def test_vech2mat3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,100)
    a = np.random.randint(10,20)

    # Generate random matrix
    A = np.random.randn(v,a*(a+1)//2,1)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D version.
    testVal = np.allclose(vech2mat3D(A)[testv,:,:], vech2mat2D(A[testv,:,:]))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: vech2mat3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `forceSym3D`. It does this by 
# generating a random example and testing against it's 2D counterpart from 
# tools2d.py.
#
# =============================================================================
def test_forceSym3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,100)
    a = np.random.randint(10,20)

    # Generate random matrix
    A = np.random.randn(v,a,a)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D version.
    testVal = np.allclose(forceSym3D(A)[testv,:,:], forceSym2D(A[testv,:,:]))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: forceSym3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `ssr3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from tools2d.py.
#
# =============================================================================
def test_ssr3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # First test spatially varying
    ssr_sv_test = ssr3D(YtX_sv, YtY, XtX_sv, beta)[testv,:,:]
    ssr_sv_expected = ssr2D(YtX_sv[testv,:,:], YtY[testv,:,:], XtX_sv[testv,:,:], beta[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(ssr_sv_test,ssr_sv_expected)

    # Now test non spatially varying
    ssr_nsv_test = ssr3D(YtX, YtY, XtX, beta)[testv,:,:]
    ssr_nsv_expected = ssr2D(YtX[testv,:,:], YtY[testv,:,:], XtX, beta[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(ssr_nsv_test,ssr_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: ssr3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `getDfromDict3D`. It does this by
# simulating random test data and checking whether the original D matrix can be
# reobtained from an dictionary version.
#
# =============================================================================
def test_getDfromDict3D():

    # Simulate some random data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nparams)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nparams[k]),Dinds[k]:(Dinds[k]+nparams[k])]

    # Now test against function
    D_test = getDfromDict3D(Ddict,nparams,nlevels)

    # Check against 2D version.
    testVal = np.allclose(D_test,D)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: getDfromDict3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)

# =============================================================================
#
# The below function tests the function `initBeta3D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_initBeta3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # First test spatially varying
    initBeta_sv_test = initBeta3D(XtX_sv, XtY_sv)[testv,:,:]
    initBeta_sv_expected = initBeta2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(initBeta_sv_test,initBeta_sv_expected)

    # Now test non spatially varying
    initBeta_nsv_test = initBeta3D(XtX, XtY)[testv,:,:]
    initBeta_nsv_expected = initBeta2D(XtX[0,:,:], XtY[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(initBeta_nsv_test,initBeta_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: initBeta3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)

# =============================================================================
#
# The below function tests the function `initSigma23D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_initSigma23D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Sum of squared residuals (spatially varying usecase)
    ete_sv = ssr3D(YtX_sv, YtY, XtX_sv, beta)

    # Sum of squared residuals (non spatially varying usecase)
    ete_nsv = ssr3D(YtX, YtY, XtX, beta)

    # First test spatially varying
    initSig2_sv_test = initSigma23D(ete_sv, n)[testv]
    initSig2_sv_expected = initSigma22D(ete_sv[testv,:,:], n)

    # Check if results are all close.
    sv_testVal = np.allclose(initSig2_sv_test,initSig2_sv_expected)

    # Now test non spatially varying
    initSig2_nsv_test = initSigma23D(ete_nsv, n)[testv]
    initSig2_nsv_expected = initSigma22D(ete_nsv[testv,:,:], n)

    # Check if results are all close.
    nsv_testVal = np.allclose(initSig2_nsv_test,initSig2_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: initSigma23D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `initDk3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from tools2d.py.
#
# =============================================================================
def test_initDk3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out the inverse duplication matrices we need.
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):
      
      invDupMatdict[i] = invDupMat2D(nparams[i]).toarray()

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Decide on a random factor
    k = np.random.randint(0,nparams.shape[0])

    # First test spatially varying
    initDk_sv_test = initDk3D(k, ZtZ_sv, Zte_sv, sigma2, nlevels, nparams, invDupMatdict)[testv,:,:]
    initDk_sv_expected = initDk2D(k, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], nlevels, nparams, invDupMatdict)
    
    # Check if results are all close.
    sv_testVal = np.allclose(initDk_sv_test,initDk_sv_expected)

    # Now test non spatially varying
    initDk_nsv_test = initDk3D(k, ZtZ, Zte, sigma2, nlevels, nparams, invDupMatdict)[testv,:,:]
    initDk_nsv_expected = initDk2D(k, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], nlevels, nparams, invDupMatdict)
    
    # Check if results are all close.
    nsv_testVal = np.allclose(initDk_nsv_test,initDk_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: initDk3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `makeDnnd3D`. It does this by 
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_makeDnnd3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Project D to be nnd
    Dnnd_test = makeDnnd3D(D)[testv,:,:]

    # Check against 2D
    Dnnd_expected = makeDnnd2D(D[testv,:,:])

    # Check if results are all close.
    testVal = np.allclose(Dnnd_test,Dnnd_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: makeDnnd3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `llh3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from tools2d.py.
#
# =============================================================================
def test_llh3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    n = Y.shape[1]
    q = Z.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Work out ete
    ete = ssr3D(YtX, YtY, XtX, beta)
    ete_sv = ssr3D(YtX_sv, YtY, XtX_sv, beta) 

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # First test spatially varying
    llh_sv_test = llh3D(n, ZtZ_sv, Zte_sv, ete_sv, sigma2, DinvIplusZtZD_sv,D)[testv]
    llh_sv_expected = llh2D(n, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], ete_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:],D[testv,:,:])[0,0]
    
    # Check if results are all close.
    sv_testVal = np.allclose(llh_sv_test,llh_sv_expected)

    # Now test non spatially varying
    llh_nsv_test = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[testv]
    llh_nsv_expected = llh2D(n, ZtZ[0,:,:], Zte[testv,:,:], ete[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:],D[testv,:,:])[0,0]
    # Check if results are all close.
    nsv_testVal = np.allclose(llh_nsv_test,llh_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: llh3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_dldB3D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_get_dldB3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    n = Y.shape[1]
    q = Z.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Work out Z'e
    Xte = XtY - XtX @ beta
    Xte_sv = XtY_sv - XtX_sv @ beta 

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # First test spatially varying
    dldB_sv_test = get_dldB3D(sigma2, Xte_sv, XtZ_sv, DinvIplusZtZD_sv, Zte_sv)[testv,:,:]
    dldB_sv_expected = get_dldB2D(sigma2[testv], Xte_sv[testv,:,:], XtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], Zte_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldB_sv_test,dldB_sv_expected)

    # Now test non spatially varying
    dldB_nsv_test = get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte)[testv,:,:]
    dldB_nsv_expected = get_dldB2D(sigma2[testv], Xte[testv,:,:], XtZ[0,:,:], DinvIplusZtZD[testv,:,:], Zte[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldB_nsv_test,dldB_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dldB3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_dldsigma23D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_get_dldsigma23D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    n = Y.shape[1]
    q = Z.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Work out ete
    ete = ssr3D(YtX, YtY, XtX, beta)
    ete_sv = ssr3D(YtX_sv, YtY, XtX_sv, beta) 

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # First test spatially varying
    dldsigma2_sv_test = get_dldsigma23D(n_sv, ete_sv, Zte_sv, sigma2, DinvIplusZtZD_sv)[testv]
    dldsigma2_sv_expected = get_dldsigma22D(n_sv[testv,:], ete_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldsigma2_sv_test,dldsigma2_sv_expected)

    # Now test non spatially varying
    dldsigma2_nsv_test = get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD)[testv]
    dldsigma2_nsv_expected = get_dldsigma22D(n, ete[testv,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldsigma2_nsv_test,dldsigma2_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dldsigma23D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_dldDk3D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_get_dldDk3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    n = Y.shape[1]
    q = Z.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Decide on a random factor
    k = np.random.randint(0,nparams.shape[0])

    # First test spatially varying
    dldDk_sv_test = get_dldDk3D(k, nlevels, nparams, ZtZ_sv, Zte_sv, sigma2, DinvIplusZtZD_sv)[testv,:,:]
    dldDk_sv_expected = get_dldDk2D(k, nlevels, nparams, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])[0]

    # Check if results are all close.
    sv_testVal = np.allclose(dldDk_sv_test,dldDk_sv_expected)

    # Now test non spatially varying
    dldDk_nsv_test = get_dldDk3D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD)[testv,:,:]
    dldDk_nsv_expected = get_dldDk2D(k, nlevels, nparams, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(dldDk_nsv_test,dldDk_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dldDk3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_covdldbeta3D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_get_covdldbeta3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    q = Z.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # First test spatially varying
    covdldB_sv_test = get_covdldbeta3D(XtZ_sv, XtX_sv, ZtZ_sv, DinvIplusZtZD_sv, sigma2)[testv,:,:]
    covdldB_sv_expected = get_covdldbeta2D(XtZ_sv[testv,:,:], XtX_sv[testv,:,:], ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], sigma2[testv])

    # Check if results are all close.
    sv_testVal = np.allclose(covdldB_sv_test,covdldB_sv_expected)

    # Now test non spatially varying
    covdldB_nsv_test = get_covdldbeta3D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2)[testv,:,:]
    covdldB_nsv_expected = get_covdldbeta2D(XtZ[0,:,:], XtX[0,:,:], ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], sigma2[testv])

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldB_nsv_test,covdldB_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_covdldbeta3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_covdldDkdsigma23D`. It does this
# by simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_get_covdldDkdsigma23D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    q = Z.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Decide on a random factor
    k = np.random.randint(0,nparams.shape[0])

    # Work out the inverse duplication matrices we need.
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):
      
      invDupMatdict[i] = invDupMat2D(nparams[i]).toarray()

    # First test spatially varying
    covdldDsigma2_sv_test = get_covdldDkdsigma23D(k, sigma2, nlevels, nparams, ZtZ_sv, DinvIplusZtZD_sv, invDupMatdict)[testv,:,:]
    covdldDsigma2_sv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nparams, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], invDupMatdict)

    # Check if results are all close.
    sv_testVal = np.allclose(covdldDsigma2_sv_test,covdldDsigma2_sv_expected)

    # Now test non spatially varying
    covdldDsigma2_nsv_test = get_covdldDkdsigma23D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict)[testv,:,:]
    covdldDsigma2_nsv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nparams, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], invDupMatdict)

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldDsigma2_nsv_test,covdldDsigma2_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_covdldDkdsigma23D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_covdldDk1Dk23D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_get_covdldDk1Dk23D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    q = Z.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Decide on 2 random factors
    k1 = np.random.randint(0,nparams.shape[0])
    k2 = np.random.randint(0,nparams.shape[0])

    # Work out the inverse duplication matrices we need.
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):
      
      invDupMatdict[i] = invDupMat2D(nparams[i]).toarray()

    # First test spatially varying
    covdldD_sv_test = get_covdldDk1Dk23D(k1, k2, nlevels, nparams, ZtZ_sv, DinvIplusZtZD_sv, invDupMatdict)[testv,:,:]
    covdldD_sv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], invDupMatdict)[0]
  
    # Check if results are all close.
    sv_testVal = np.allclose(covdldD_sv_test,covdldD_sv_expected)

    # Now test non spatially varying
    covdldD_nsv_test = get_covdldDk1Dk23D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict)[testv,:,:]
    covdldD_nsv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], invDupMatdict)[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(covdldD_nsv_test,covdldD_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_covdldDk1Dk23D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `getCovergedIndices`. It does this by
# testing against a predefined example.
#
# =============================================================================
def test_getCovergedIndices():

    # Suppose we were looking at 20 voxels.
    convergedBeforeIt = np.zeros(20)

    # Suppose we've done a few iterations already and during those iterations
    # we saw voxels 0-5 and 10-15 converge
    convergedBeforeIt[0:5] = 1
    convergedBeforeIt[10:15] = 1

    # Now we have 10 voxels left to consider.
    convergedDuringIt = np.zeros(10)

    # Lets suppose in our latest iteration that the first 5 of the remaining
    # voxels (voxels 5-10 of our original list) converge.
    convergedDuringIt[0:5]=1

    # We now expect the voxels which have converged to be voxels 0-15
    indices_ConAfterIt_expected = np.arange(15)

    # And we expect the voxels which have not converged to be 15-20
    indices_notConAfterIt_expected = np.arange(15,20)

    # The voxels which converged during the last iteration were voxels 5-10
    indices_conDuringIt_expected = np.arange(5,10)

    # Of the 10 voxels we looked at during the iteration the first 5 converged
    local_converged_expected = np.arange(5)

    # Of the 10 voxels we looked at during the iteration the last 5 didn't
    # converge
    local_notconverged_expected = np.arange(5,10)

    # Lets see if the function managed the same.
    indices_ConAfterIt_test, indices_notConAfterIt_test, indices_conDuringIt_test, local_converged_test, local_notconverged_test = getConvergedIndices(convergedBeforeIt, convergedDuringIt)

    # Check each outcome is as expected
    testVal = np.allclose(indices_ConAfterIt_expected,indices_ConAfterIt_test)
    testVal = testVal and np.allclose(indices_notConAfterIt_expected,indices_notConAfterIt_test)
    testVal = testVal and np.allclose(indices_conDuringIt_expected,indices_conDuringIt_test)
    testVal = testVal and np.allclose(local_converged_expected,local_converged_test)
    testVal = testVal and np.allclose(local_notconverged_expected,local_notconverged_test)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: getCovergedIndices')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `block2stacked3D`. It does this by
# generating a random example and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_block2stacked3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,140)
    n1 = np.random.randint(10,20)
    n2 = np.random.randint(10,20)
    l1 = np.random.randint(50,100)
    l2 = np.random.randint(50,100)

    # Work out m1 and m2
    m1 = n1*l1
    m2 = n2*l2

    # Generate random A
    A = np.random.randn(v,m1,m2)

    # Save partition
    pA = np.array([n1,n2])

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D counterpart
    testVal = np.allclose(block2stacked3D(A,pA)[testv,:,:],block2stacked2D(A[testv,:,:],pA))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: block2stacked3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    return(result)


# =============================================================================
#
# The below function tests the function `mat2vecb3D`. It does this by
# generating a random example and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_mat2vecb3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,140)
    n1 = np.random.randint(10,20)
    n2 = np.random.randint(10,20)
    l1 = np.random.randint(50,100)
    l2 = np.random.randint(50,100)

    # Work out m1 and m2
    m1 = n1*l1
    m2 = n2*l2

    # Generate random A
    A = np.random.randn(v,m1,m2)

    # Save partition
    pA = np.array([n1,n2])

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D counterpart
    testVal = np.allclose(mat2vecb3D(A,pA)[testv,:,:],mat2vecb2D(A[testv,:,:],pA))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mat2vecb3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    return(result)


# =============================================================================
#
# The below function tests the function `sumAijBijt3D`. It does this by
# generating a random example and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_sumAijBijt3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,140)
    n1 = np.random.randint(2,10)
    n1prime = np.random.randint(2,10)
    n2 = np.random.randint(2,10)
    l1 = np.random.randint(50,100)
    l2 = np.random.randint(50,100)

    # Work out m1 and m2
    m1 = n1*l1
    m1prime = n1prime*l1
    m2 = n2*l2

    # Generate random A and B
    A = np.random.randn(v,m1,m2)
    B = np.random.randn(v,m1prime,m2)

    # Save partition
    pA = np.array([n1,n2])
    pB = np.array([n1prime,n2])

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D counterpart
    testVal = np.allclose(sumAijBijt3D(A, B, pA, pB)[testv,:,:],sumAijBijt2D(A[testv,:,:], B[testv,:,:], pA, pB))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: sumAijBijt3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    return(result)

# =============================================================================
#
# The below function tests the function `sumAijBijt3D`. It does this by
# generating a random example and testing against it's 2D counterpart from
# tools2d.py.
#
# =============================================================================
def test_sumAijKronBij3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,140)
    n1 = np.random.randint(2,10)
    n2 = np.random.randint(2,10)
    l1 = np.random.randint(50,100)
    l2 = np.random.randint(50,100)

    # Work out m1 and m2
    m1 = n1*l1
    m2 = n2*l2

    # Generate random A and B
    A = np.random.randn(v,m1,m2)
    B = np.random.randn(v,m1,m2)

    # Save partition
    p = np.array([n1,n2])

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Check against 2D counterpart
    testVal = np.allclose(sumAijKronBij3D(A, B, p)[0][testv,:,:],sumAijKronBij2D(A[testv,:,:], B[testv,:,:], p)[0])

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: sumAijKronBij3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    return(result)


# =============================================================================
#
# The below function tests the function `resms3D`. It does this by simulating
# random test data and testing against niave computation using `ssr2D` from 
# tools2d.py.
#
# =============================================================================
def test_resms3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData()
    v = Y.shape[0]
    n = X.shape[0]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # First test spatially varying
    resms_sv_test = resms3D(YtX_sv, YtY, XtX_sv, beta, n_sv, p)[testv,:,:]
    resms_sv_expected = ssr2D(YtX_sv[testv,:,:], YtY[testv,:,:], XtX_sv[testv,:,:], beta[testv,:,:])/(n_sv[testv,:,:]-p)

    # Check if results are all close.
    sv_testVal = np.allclose(resms_sv_test,resms_sv_expected)

    # Now test non spatially varying
    resms_nsv_test = resms3D(YtX, YtY, XtX, beta, n, p)[testv,:,:]
    resms_nsv_expected = ssr2D(YtX[testv,:,:], YtY[testv,:,:], XtX, beta[testv,:,:])/(n-p)

    # Check if results are all close.
    nsv_testVal = np.allclose(resms_nsv_test,resms_nsv_expected)

    # Check against 2D version.
    testVal = nsv_testVal and sv_testVal

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: resms3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)
    

# =============================================================================
#
# The below function tests the function `covB3D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_covB3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))
    p = X.shape[1]
    n = X.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta[testv,:,:]

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D[testv,:,:] @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB_expected = np.linalg.inv(X.transpose() @ invV @ X/sigma2[testv])

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    covB_test = get_covB3D(XtX, XtZ, DinvIplusZtZD, sigma2)[testv,:]

    # Check if results are all close.
    testVal = np.allclose(covB_test,covB_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: covB3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `varLB3D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_varLB3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))
    p = X.shape[1]
    n = X.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta[testv,:,:]

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D[testv,:,:] @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB = np.linalg.inv(X.transpose() @ invV @ X/sigma2[testv])
    
    # Variance of L times beta
    varLB_expected = L @ covB @ L.transpose()

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    varLB_test = get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2)[testv,:]

    # Check if results are all close.
    testVal = np.allclose(varLB_test,varLB_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: varLB3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_T3D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_get_T3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))
    p = X.shape[1]
    n = X.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta[testv,:,:]

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D[testv,:,:] @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB = np.linalg.inv(X.transpose() @ invV @ X/sigma2[testv])
    
    # Variance of L times beta
    varLB = L @ covB @ L.transpose()

    # Work out T
    T_expected = LB/np.sqrt(varLB) 

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    T_test = get_T3D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2)[testv,:]

    # Check if results are all close.
    testVal = np.allclose(T_test,T_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_T3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_F3D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_get_F3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))
    p = X.shape[1]
    n = X.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.eye(p)
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta[testv,:,:]

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D[testv,:,:] @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB = np.linalg.inv(X.transpose() @ invV @ X/sigma2[testv])
    
    # Variance of L times beta
    varLB = L @ covB @ L.transpose()

    # Work out F
    F_expected = LB.transpose() @ np.linalg.inv(varLB) @ LB/rL

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    F_test = get_F3D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2)[testv,:]

    # Check if results are all close.
    testVal = np.allclose(F_test,F_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_F3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_R23D`. It does this by simulating
# a random example and testing against niave calculation.
#
# =============================================================================
def test_get_R23D():

    # Random number of voxels
    v = np.random.randint(200)

    # Random "F" statistics
    F = np.random.randn(v).reshape(v,1)**2

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,np.random.randint(2,10)))
    L[0,0]=1

    # Rank of contrast vector
    rL = np.linalg.matrix_rank(L)

    # Random degrees of freedom
    df_denom = np.random.binomial(100,0.7,size=(v,1))+1

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Expected R^2
    R2_expected = (rL*F[testv,:])/(rL*F[testv,:] + df_denom[testv,:])

    # Test R^2
    R2_test = get_R23D(L, F[testv,:], df_denom[testv,:])

    # Check if results are all close.
    testVal = np.allclose(R2_test,R2_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_R23D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `T2P3D`. It does this by simulating
# a random example and testing against niave calculation.
#
# =============================================================================
def test_T2P3D():

    # Random number of voxels
    v = np.random.randint(200)

    # Random "T" statistics
    T = np.random.randn(v).reshape(v,1)

    # Random minlog value
    minlog = -np.random.randint(500,1000)

    # Random degrees of freedom
    df = np.random.binomial(100,0.7,size=(v,1))+1

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Expected P value
    P_expected = -np.log10(1-stats.t.cdf(T[testv,:], df[testv,:]))

    # Remove infs
    if np.isinf(P_expected) and P_expected<0:

        P_expected = minlog

    P_test = T2P3D(T,df,minlog)[testv,:]

    # Check if results are all close.
    testVal = np.allclose(P_test,P_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: T2P3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `F2P3D`. It does this by simulating
# a random example and testing against niave calculation.
#
# =============================================================================
def test_F2P3D():

    # Random number of voxels
    v = np.random.randint(200)

    # Random "F" statistics
    F = np.random.randn(v).reshape(v,1)**2

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,np.random.randint(2,10)))
    L[0,0]=1

    # Random minlog value
    minlog = -np.random.randint(500,1000)

    # Random degrees of freedom
    df_denom = np.random.binomial(100,0.7,size=(v,1))+1

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Expected P value
    P_expected = -np.log10(1-stats.f.cdf(F[testv,:], np.linalg.matrix_rank(L), df_denom[testv,:]))

    # Remove infs
    if np.isinf(P_expected) and P_expected<0:

        P_expected = minlog

    P_test = F2P3D(F, L, df_denom, minlog)[testv,:]

    # Check if results are all close.
    testVal = np.allclose(P_test,P_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: F2P3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_swdf_T3D`. It does this by
# simulating random test data and testing against niave calculation.
#
# Note: This test assumes the correctness of the functions `get_InfoMat3D` and
# `get_dS2`.
#
# =============================================================================
def test_get_swdf_T3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))
    p = X.shape[1]
    n = X.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Get D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2)[testv,:,:]
    
    # Get derivative of S^2
    dS2 = get_dS23D(nparams, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2)[testv,:,:]

    # Get Fisher information matrix
    InfoMat = get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nparams, ZtZ)[testv,:,:]

    # Calculate df estimator
    swdf_expected = 2*(S2**2)/(dS2.transpose() @ np.linalg.inv(InfoMat) @ dS2)


    swdf_test = get_swdf_T3D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nparams)[testv,:,:]

    # Check if results are all close.
    testVal = np.allclose(swdf_test,swdf_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_swdf_T3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_swdf_F3D`. It does this by
# simulating random test data and testing against niave calculation.
#
# Note: This test assumes the correctness of the functions `get_swdf_T3D`.
#
# =============================================================================
def test_get_swdf_F3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))
    p = X.shape[1]
    n = X.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # L is rL in rank
    rL = np.linalg.matrix_rank(L)

    # Initialize empty sum.
    sum_swdf_adj = 0

    # Loop through first rL rows of L
    for i in np.arange(rL):

        # Work out the swdf for each row of L
        swdf_row = get_swdf_T3D(L[i:(i+1),:], D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nparams)[testv,:,:]

        # Work out adjusted df = df/(df-2)
        swdf_adj = swdf_row/(swdf_row-2)

        # Add to running sum
        sum_swdf_adj = sum_swdf_adj + swdf_adj[0]

    # Work out final df
    swdf_expected = 2*sum_swdf_adj/(sum_swdf_adj-rL)

    # Function version 
    swdf_test = get_swdf_F3D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nparams)[testv]

    # Check if results are all close.
    testVal = np.allclose(swdf_test,swdf_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_swdf_F3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_dS23D`. It does this by
# simulating random test data and testing against niave calculation.
#
# =============================================================================
def test_get_dS23D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))
    p = X.shape[1]
    n = X.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Niave calculation for one voxel.
    # ----------------------------------------------------------------------------------
    IplusZDZt = np.eye(n) + Z @ D[testv,:,:] @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    # Calculate X'V^{-1}X
    XtiVX = X.transpose() @ invV @ X 

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2_expected = np.zeros((1+np.int32(np.sum(nparams*(nparams+1)/2)),1))

    # Work out indices for each start of each component of vector 
    # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
    DerivInds = np.int32(np.cumsum(nparams*(nparams+1)/2) + 1)
    DerivInds = np.insert(DerivInds,0,1)

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.inv(XtiVX) @ L.transpose()

    # Add to dS2
    dS2_expected[0:1] = dS2dsigma2.reshape(dS2_expected[0:1].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nparams)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((np.int32(nparams[k]*(nparams[k]+1)/2),1))

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nparams)
                    
            # Work out Z_(k,j)'
            Zkjt = Z[:,Ikj].transpose()

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = Zkjt @ invV @ X

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.inv(XtiVX) @ L.transpose()
            
            # Sum terms
            dS2dvechDk = dS2dvechDk + mat2vech2D(np.kron(K,K.transpose()))

        # Multiply by sigma^2
        dS2dvechDk = sigma2[testv]*dS2dvechDk

        # Add to dS2
        dS2_expected[DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2_expected[DerivInds[k]:DerivInds[k+1]].shape)


    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Obtain result from function
    dS2_test = get_dS23D(nparams, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2)[testv,:,:]

    # Check if results are all close.
    testVal = np.allclose(dS2_test,dS2_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dS23D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_InfoMat3D`. It does this by
# simulating random test data and testing against niave calculation.
#
# =============================================================================
def test_get_InfoMat3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData(v=10)
    v = Y.shape[0]
    q = np.sum(np.dot(nparams,nlevels))

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats(Y,Z,X,Z_sv,X_sv)
    n = X.shape[0]

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

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
    FI_expected = np.zeros((tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2[testv]**2))
    
    # Add dl/dsigma2 covariance
    FI_expected[0,0] = covdldsigma2

    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nparams)):

        # Get covariance of dldsigma and dldD      
        covdldsigmadD = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nparams, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], invDupMatdict)[0].reshape(FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FI_expected[0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
        FI_expected[FishIndsDk[k]:FishIndsDk[k+1],0:1] = FI_expected[0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose()
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nparams)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], invDupMatdict)[0]

            # Add to FImat
            FI_expected[np.ix_(IndsDk1, IndsDk2)] = covdldDk1dDk2
            FI_expected[np.ix_(IndsDk2, IndsDk1)] = FI_expected[np.ix_(IndsDk1, IndsDk2)].transpose()

    FI_test = get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nparams, ZtZ)[testv,:,:]

    # Check if results are all close.
    testVal = np.allclose(FI_test,FI_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_InfoMat3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function runs all unit tests and outputs the results.
#
# =============================================================================
def run_all3D():

    # Record passed and failed tests.
    passedTests = np.array([])
    failedTests = np.array([])

    # Test kron3D
    name = 'kron3D'
    result = test_kron3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test mat2vec3D
    name = 'mat2vec3D'
    result = test_mat2vec3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test mat2vech3D
    name = 'mat2vech3D'
    result = test_mat2vech3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test vec2mat3D
    name = 'vec2mat3D'
    result = test_vec2mat3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test vech2mat3D
    name = 'vech2mat3D'
    result = test_vech2mat3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test forceSym3D
    name = 'forceSym3D'
    result = test_forceSym3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test ssr3D
    name = 'ssr3D'
    result = test_ssr3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test getDfromDict3D
    name = 'getDfromDict3D'
    result = test_getDfromDict3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test initBeta3D
    name = 'initBeta3D'
    result = test_initBeta3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test initSigma23D
    name = 'initSigma23D'
    result = test_initSigma23D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test initDk3D
    name = 'initDk3D'
    result = test_initDk3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test makeDnnd3D
    name = 'makeDnnd3D'
    result = test_makeDnnd3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test llh3D
    name = 'llh3D'
    result = test_llh3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test get_dldB3D
    name = 'get_dldB3D'
    result = test_get_dldB3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test get_dldsigma23D
    name = 'get_dldsigma23D'
    result = test_get_dldsigma23D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test get_dldDk3D
    name = 'get_dldDk3D'
    result = test_get_dldDk3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test get_covdldbeta3D
    name = 'get_covdldbeta3D'
    result = test_get_covdldbeta3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test get_covdldDkdsigma23D
    name = 'get_covdldDkdsigma23D'
    result = test_get_covdldDkdsigma23D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test get_covdldDk1Dk23D
    name = 'get_covdldDk1Dk23D'
    result = test_get_covdldDk1Dk23D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test getCovergedIndices
    name = 'getCovergedIndices'
    result = test_getCovergedIndices()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test block2stacked3D
    name = 'block2stacked3D'
    result = test_block2stacked3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test mat2vecb3D
    name = 'mat2vecb3D'
    result = test_mat2vecb3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test sumAijBijt3D
    name = 'sumAijBijt3D'
    result = test_sumAijBijt3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test sumAijKronBij3D
    name = 'sumAijKronBij3D'
    result = test_sumAijKronBij3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test covB3D
    name = 'covB3D'
    result = test_covB3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test varLB3D
    name = 'varLB3D'
    result = test_varLB3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_T3D
    name = 'get_T3D'
    result = test_get_T3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_F3D
    name = 'get_F3D'
    result = test_get_F3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_R23D
    name = 'get_R23D'
    result = test_get_R23D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test T2P3D
    name = 'T2P3D'
    result = test_T2P3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test F2P3D
    name = 'F2P3D'
    result = test_F2P3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_swdf_T3D
    name = 'get_swdf_T3D'
    result = test_get_swdf_T3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_swdf_F3D
    name = 'get_swdf_F3D'
    result = test_get_swdf_F3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_dS23D
    name = 'get_dS23D'
    result = test_get_dS23D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_InfoMat3D
    name = 'get_InfoMat3D'
    result = test_get_InfoMat3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    print('=============================================================')

    print('Tests completed')
    print('-------------------------------------------------------------')
    print('Summary:')
    print('Passed Tests: ', passedTests)
    print('Failed Tests: ', failedTests)
