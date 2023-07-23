import os
import sys
import numpy as np
from scipy import stats

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.npMatrix2d import *
from lib.npMatrix3d import *
from genTestDat import prodMats3D, genTestData3D
# =============================================================================
# This file contains all unit tests for the functions given in the npMatrix3D.py
# file.
#
# Author: Tom Maullin
# Last edited: 22/03/2020su
#
# =============================================================================

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
# a random example and testing against it's 2D counterpart from npMatrix2d.py.
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
# npMatrix2d.py.
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
# npMatrix2d.py.
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
# npMatrix2d.py.
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
# npMatrix2d.py.
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
# random test data and testing against it's 2D counterpart from npMatrix2d.py.
#
# =============================================================================
def test_ssr3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D()
    v = Y.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D()

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Now test against function
    D_test = getDfromDict3D(Ddict,nraneffs,nlevels)

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
# npMatrix2d.py.
#
# =============================================================================
def test_initBeta3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D()
    v = Y.shape[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
# npMatrix2d.py.
#
# =============================================================================
def test_initSigma23D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D()
    v = Y.shape[0]
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
# random test data and testing against it's 2D counterpart from npMatrix2d.py.
#
# =============================================================================
def test_initDk3D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # First test spatially varying
    initDk_sv_test = initDk3D(k, ZtZ_diag_sv, Zte_sv, sigma2, nlevels, nraneffs, dupMatTdict)[testv,:,:]
    initDk_sv_expected = initDk2D(k, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], nlevels, nraneffs, dupMatTdict)
    
    # Check if results are all close.
    sv_testVal = np.allclose(initDk_sv_test,initDk_sv_expected)

    # Now test non spatially varying
    initDk_nsv_test = initDk3D(k, ZtZ_diag, Zte, sigma2, nlevels, nraneffs, dupMatTdict)[testv,:,:]
    initDk_nsv_expected = initDk2D(k, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], nlevels, nraneffs, dupMatTdict)
    
    # Check if results are all close.
    nsv_testVal = np.allclose(initDk_nsv_test,initDk_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Flattened ZtZ
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)

    # Flattened ZtZ_sv
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # First test spatially varying
    initDk_sv_test = initDk3D(k, ZtZ_sv_flattened, Zte_sv, sigma2, nlevels, nraneffs, dupMatTdict)[testv,:,:]
    initDk_sv_expected = initDk2D(k, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], nlevels, nraneffs, dupMatTdict)
    
    # Check if results are all close.
    sv_testVal = np.allclose(initDk_sv_test,initDk_sv_expected)

    # Now test non spatially varying
    initDk_nsv_test = initDk3D(k, ZtZ_flattened, Zte, sigma2, nlevels, nraneffs, dupMatTdict)[testv,:,:]
    initDk_nsv_expected = initDk2D(k, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], nlevels, nraneffs, dupMatTdict)
    
    # Check if results are all close.
    nsv_testVal = np.allclose(initDk_nsv_test,initDk_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # First test spatially varying
    initDk_sv_test = initDk3D(k, ZtZ_sv, Zte_sv, sigma2, nlevels, nraneffs, dupMatTdict)[testv,:,:]
    initDk_sv_expected = initDk2D(k, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], nlevels, nraneffs, dupMatTdict)
    
    # Check if results are all close.
    sv_testVal = np.allclose(initDk_sv_test,initDk_sv_expected)

    # Now test non spatially varying
    initDk_nsv_test = initDk3D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict)[testv,:,:]
    initDk_nsv_expected = initDk2D(k, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], nlevels, nraneffs, dupMatTdict)
    
    # Check if results are all close.
    nsv_testVal = np.allclose(initDk_nsv_test,initDk_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine test results
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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
# The below function tests the function `get_DinvIplusZtZD3D`. It does this by 
# simulating random test data and testing against niave calculation
#
# =============================================================================
def test_get_DinvIplusZtZD3D():
    
    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)

    # Generate product matrices (full versions)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_sv_diag = np.einsum('ijj->ij', ZtZ_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]
        
    # Expected result (nsv)
    DinvIplusZtZD_nsv_expected = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # Test result (nsv)
    DinvIplusZtZD_diag_nsv_test = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    # Expected result (sv)
    DinvIplusZtZD_sv_expected = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ_sv, D))

    # Test result (sv)
    DinvIplusZtZD_diag_sv_test = get_DinvIplusZtZD3D(Ddict, None, ZtZ_sv_diag, nlevels, nraneffs) 

    # Convert Test results to non-diagonal form
    DinvIplusZtZD_sv_test = np.zeros((v,q,q))
    DinvIplusZtZD_nsv_test = np.zeros((v,q,q))
    np.einsum('ijj->ij', DinvIplusZtZD_sv_test)[...] = DinvIplusZtZD_diag_sv_test
    np.einsum('ijj->ij', DinvIplusZtZD_nsv_test)[...] = DinvIplusZtZD_diag_nsv_test

    # Check if results are all close.
    testVal_nsv = np.allclose(DinvIplusZtZD_nsv_test,DinvIplusZtZD_nsv_expected)
    testVal_sv = np.allclose(DinvIplusZtZD_sv_test,DinvIplusZtZD_sv_expected)
    testVal_tc1 = testVal_nsv and testVal_sv

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([3])
    nlevels = np.array([200])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)

    # Generate product matrices (full versions)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]
        
    # Expected result (nsv)
    DinvIplusZtZD_nsv_expected = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Flattened ZtZ
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)

    # Flattened ZtZ_sv
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Test result (nsv)
    DinvIplusZtZD_nsv_test = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 

    # Expected result (sv)
    DinvIplusZtZD_sv_expected = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ_sv, D))

    # Test result (sv)
    DinvIplusZtZD_sv_test = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # Initial test values
    testVal_nsv = True
    testVal_sv = True

    # Loop through and check all expected blocks are equal
    for i in np.arange(l0):
        
        testVal_sv = testVal_sv and np.allclose(DinvIplusZtZD_sv_test[testv,:,i*q0:(i+1)*q0],DinvIplusZtZD_sv_expected[testv,i*q0:(i+1)*q0,i*q0:(i+1)*q0])
        testVal_nsv = testVal_nsv and np.allclose(DinvIplusZtZD_nsv_test[0,:,i*q0:(i+1)*q0],DinvIplusZtZD_nsv_expected[0,i*q0:(i+1)*q0,i*q0:(i+1)*q0])

    testVal_tc2 = testVal_nsv and testVal_sv

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    q = np.sum(nlevels*nraneffs)

    # Generate product matrices (full versions)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]
        
    # Expected result (nsv)
    DinvIplusZtZD_nsv_expected = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # Test result (nsv)
    DinvIplusZtZD_nsv_test = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs) 

    # Expected result (sv)
    DinvIplusZtZD_sv_expected = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ_sv, D))

    # Test result (sv)
    DinvIplusZtZD_sv_test = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv, nlevels, nraneffs) 

    # Check if results are all close.
    testVal_nsv = np.allclose(DinvIplusZtZD_nsv_test,DinvIplusZtZD_nsv_expected)
    testVal_sv = np.allclose(DinvIplusZtZD_sv_test,DinvIplusZtZD_sv_expected)
    testVal_tc3 = testVal_nsv and testVal_sv

    # Combine test values from all cases
    testVal = testVal_tc1 and testVal_tc2 and testVal_tc3

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_DinvIplusZtZD3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)

# =============================================================================
#
# The below function tests the function `makeDnnd3D`. It does this by 
# simulating random test data and testing against it's 2D counterpart from
# npMatrix2d.py.
#
# =============================================================================
def test_makeDnnd3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D()
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
# random test data and testing against it's 2D counterpart from npMatrix2d.py.
#
# =============================================================================
def test_llh3D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------
    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # Get D dict
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)
    Ddict = dict()
    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):
        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # First test spatially varying
    llh_sv_test = llh3D(n, ZtZ_diag_sv, Zte_sv, ete_sv, sigma2, DinvIplusZtZD_diag_sv,D, Ddict, nlevels, nraneffs)[testv]
    llh_sv_expected = llh2D(n, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], ete_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:],D[testv,:,:])[0,0]
    
    # Check if results are all close.
    sv_testVal = np.allclose(llh_sv_test,llh_sv_expected)

    # Now test non spatially varying
    llh_nsv_test = llh3D(n, ZtZ_diag, Zte, ete, sigma2, DinvIplusZtZD_diag,D, Ddict, nlevels, nraneffs)[testv]
    llh_nsv_expected = llh2D(n, ZtZ[0,:,:], Zte[testv,:,:], ete[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:],D[testv,:,:])[0,0]
    # Check if results are all close.
    nsv_testVal = np.allclose(llh_nsv_test,llh_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Work out ete
    ete = ssr3D(YtX, YtY, XtX, beta)
    ete_sv = ssr3D(YtX_sv, YtY, XtX_sv, beta) 

    # Get D dict
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)
    Ddict = dict()
    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):
        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) 
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Obtain D(I+Z'ZD)^(-1) flattened
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 
    DinvIplusZtZD_sv_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # First test spatially varying
    llh_sv_test = llh3D(n, ZtZ_sv_flattened, Zte_sv, ete_sv, sigma2, DinvIplusZtZD_sv_flattened,D, Ddict, nlevels, nraneffs)[testv]
    llh_sv_expected = llh2D(n, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], ete_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:],D[testv,:,:])[0,0]
    
    # Check if results are all close.
    sv_testVal = np.allclose(llh_sv_test,llh_sv_expected)

    # Now test non spatially varying
    llh_nsv_test = llh3D(n, ZtZ_flattened, Zte, ete, sigma2, DinvIplusZtZD_flattened,D, Ddict, nlevels, nraneffs)[testv]
    llh_nsv_expected = llh2D(n, ZtZ[0,:,:], Zte[testv,:,:], ete[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:],D[testv,:,:])[0,0]

    # Check if results are all close.
    nsv_testVal = np.allclose(llh_nsv_test,llh_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Work out ete
    ete = ssr3D(YtX, YtY, XtX, beta)
    ete_sv = ssr3D(YtX_sv, YtY, XtX_sv, beta) 

    # Get D dict
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)
    Ddict = dict()
    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):
        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs) 
    DinvIplusZtZD_sv = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv, nlevels, nraneffs) 

    # First test spatially varying
    llh_sv_test = llh3D(n, ZtZ_sv, Zte_sv, ete_sv, sigma2, DinvIplusZtZD_sv,D, Ddict, nlevels, nraneffs)[testv]
    llh_sv_expected = llh2D(n, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], ete_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:],D[testv,:,:])[0,0]
    
    # Check if results are all close.
    sv_testVal = np.allclose(llh_sv_test,llh_sv_expected)

    # Now test non spatially varying
    llh_nsv_test = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D, Ddict, nlevels, nraneffs)[testv]
    llh_nsv_expected = llh2D(n, ZtZ[0,:,:], Zte[testv,:,:], ete[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:],D[testv,:,:])[0,0]
    # Check if results are all close.
    nsv_testVal = np.allclose(llh_nsv_test,llh_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine results
    testVal = testVal_tc1 and testVal_tc2 and testVal_tc3

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
# npMatrix2d.py.
#
# =============================================================================
def test_get_dldB3D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # First test spatially varying
    dldB_sv_test = get_dldB3D(sigma2, Xte_sv, XtZ_sv, DinvIplusZtZD_diag_sv, Zte_sv, nraneffs)[testv,:,:]
    dldB_sv_expected = get_dldB2D(sigma2[testv], Xte_sv[testv,:,:], XtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], Zte_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldB_sv_test,dldB_sv_expected)

    # Now test non spatially varying
    dldB_nsv_test = get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD_diag, Zte, nraneffs)[testv,:,:]
    dldB_nsv_expected = get_dldB2D(sigma2[testv], Xte[testv,:,:], XtZ[0,:,:], DinvIplusZtZD[testv,:,:], Zte[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldB_nsv_test,dldB_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal


    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Work out Z'e
    Xte = XtY - XtX @ beta
    Xte_sv = XtY_sv - XtX_sv @ beta 

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Obtain D(I+Z'ZD)^(-1) flattened
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 
    DinvIplusZtZD_sv_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # First test spatially varying
    dldB_sv_test = get_dldB3D(sigma2, Xte_sv, XtZ_sv, DinvIplusZtZD_sv_flattened, Zte_sv, nraneffs)[testv,:,:]
    dldB_sv_expected = get_dldB2D(sigma2[testv], Xte_sv[testv,:,:], XtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], Zte_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldB_sv_test,dldB_sv_expected)

    # Now test non spatially varying
    dldB_nsv_test = get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD_flattened, Zte, nraneffs)[testv,:,:]
    dldB_nsv_expected = get_dldB2D(sigma2[testv], Xte[testv,:,:], XtZ[0,:,:], DinvIplusZtZD[testv,:,:], Zte[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldB_nsv_test,dldB_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
    dldB_sv_test = get_dldB3D(sigma2, Xte_sv, XtZ_sv, DinvIplusZtZD_sv, Zte_sv, nraneffs)[testv,:,:]
    dldB_sv_expected = get_dldB2D(sigma2[testv], Xte_sv[testv,:,:], XtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], Zte_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldB_sv_test,dldB_sv_expected)

    # Now test non spatially varying
    dldB_nsv_test = get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte, nraneffs)[testv,:,:]
    dldB_nsv_expected = get_dldB2D(sigma2[testv], Xte[testv,:,:], XtZ[0,:,:], DinvIplusZtZD[testv,:,:], Zte[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldB_nsv_test,dldB_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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
# npMatrix2d.py.
#
# =============================================================================
def test_get_dldsigma23D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # First test spatially varying
    dldsigma2_sv_test = get_dldsigma23D(n_sv, ete_sv, Zte_sv, sigma2, DinvIplusZtZD_diag_sv, nraneffs)[testv]
    dldsigma2_sv_expected = get_dldsigma22D(n_sv[testv,:], ete_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldsigma2_sv_test,dldsigma2_sv_expected)

    # Now test non spatially varying
    dldsigma2_nsv_test = get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD_diag, nraneffs)[testv]
    dldsigma2_nsv_expected = get_dldsigma22D(n, ete[testv,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldsigma2_nsv_test,dldsigma2_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Obtain D(I+Z'ZD)^(-1) flattened
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 
    DinvIplusZtZD_sv_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # First test spatially varying
    dldsigma2_sv_test = get_dldsigma23D(n_sv, ete_sv, Zte_sv, sigma2, DinvIplusZtZD_sv_flattened, nraneffs)[testv]
    dldsigma2_sv_expected = get_dldsigma22D(n_sv[testv,:], ete_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldsigma2_sv_test,dldsigma2_sv_expected)

    # Now test non spatially varying
    dldsigma2_nsv_test = get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD_flattened, nraneffs)[testv]
    dldsigma2_nsv_expected = get_dldsigma22D(n, ete[testv,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldsigma2_nsv_test,dldsigma2_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
    dldsigma2_sv_test = get_dldsigma23D(n_sv, ete_sv, Zte_sv, sigma2, DinvIplusZtZD_sv, nraneffs)[testv]
    dldsigma2_sv_expected = get_dldsigma22D(n_sv[testv,:], ete_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])

    # Check if results are all close.
    sv_testVal = np.allclose(dldsigma2_sv_test,dldsigma2_sv_expected)

    # Now test non spatially varying
    dldsigma2_nsv_test = get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD, nraneffs)[testv]
    dldsigma2_nsv_expected = get_dldsigma22D(n, ete[testv,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])

    # Check if results are all close.
    nsv_testVal = np.allclose(dldsigma2_nsv_test,dldsigma2_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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
# npMatrix2d.py.
#
# =============================================================================
def test_get_dldDk3D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # First test spatially varying
    dldDk_sv_test,ZtZmat = get_dldDk3D(k, nlevels, nraneffs, ZtZ_diag_sv, Zte_sv, sigma2, DinvIplusZtZD_diag_sv)
    dldDk_sv_test,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ_diag_sv, Zte_sv, sigma2, DinvIplusZtZD_diag_sv, ZtZmat=ZtZmat)
    dldDk_sv_test = dldDk_sv_test[testv,:,:] 
    dldDk_sv_expected = get_dldDk2D(k, nlevels, nraneffs, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])[0]

    # Check if results are all close.
    sv_testVal = np.allclose(dldDk_sv_test,dldDk_sv_expected)

    # Now test non spatially varying
    dldDk_nsv_test,ZtZmat = get_dldDk3D(k, nlevels, nraneffs, ZtZ_diag, Zte, sigma2, DinvIplusZtZD_diag)
    dldDk_nsv_test,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ_diag, Zte, sigma2, DinvIplusZtZD_diag)
    dldDk_nsv_test = dldDk_nsv_test[testv,:,:] 
    dldDk_nsv_expected = get_dldDk2D(k, nlevels, nraneffs, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(dldDk_nsv_test,dldDk_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]
    
    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Obtain D(I+Z'ZD)^(-1) flattened
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 
    DinvIplusZtZD_sv_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # Factor number
    k = 0

    # First test spatially varying
    dldDk_sv_test,ZtZmat = get_dldDk3D(k, nlevels, nraneffs, ZtZ_sv_flattened, Zte_sv, sigma2, DinvIplusZtZD_sv_flattened)
    dldDk_sv_test,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ_sv_flattened, Zte_sv, sigma2, DinvIplusZtZD_sv_flattened, ZtZmat=ZtZmat)
    dldDk_sv_test = dldDk_sv_test[testv,:,:] 
    dldDk_sv_expected = get_dldDk2D(k, nlevels, nraneffs, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])[0]

    # Check if results are all close.
    sv_testVal = np.allclose(dldDk_sv_test,dldDk_sv_expected)

    # Now test non spatially varying
    dldDk_nsv_test,ZtZmat = get_dldDk3D(k, nlevels, nraneffs, ZtZ_flattened, Zte, sigma2, DinvIplusZtZD_flattened)
    dldDk_nsv_test,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ_flattened, Zte, sigma2, DinvIplusZtZD_flattened)
    dldDk_nsv_test = dldDk_nsv_test[testv,:,:] 
    dldDk_nsv_expected = get_dldDk2D(k, nlevels, nraneffs, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(dldDk_nsv_test,dldDk_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out Z'e
    Zte = ZtY - ZtX @ beta
    Zte_sv = ZtY_sv - ZtX_sv @ beta

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # First test spatially varying
    dldDk_sv_test,ZtZmat = get_dldDk3D(k, nlevels, nraneffs, ZtZ_sv, Zte_sv, sigma2, DinvIplusZtZD_sv)
    dldDk_sv_test,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ_sv, Zte_sv, sigma2, DinvIplusZtZD_sv, ZtZmat=ZtZmat)
    dldDk_sv_test = dldDk_sv_test[testv,:,:] 
    dldDk_sv_expected = get_dldDk2D(k, nlevels, nraneffs, ZtZ_sv[testv,:,:], Zte_sv[testv,:,:], sigma2[testv], DinvIplusZtZD_sv[testv,:,:])[0]

    # Check if results are all close.
    sv_testVal = np.allclose(dldDk_sv_test,dldDk_sv_expected)

    # Now test non spatially varying
    dldDk_nsv_test,ZtZmat = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD)
    dldDk_nsv_test,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD)
    dldDk_nsv_test = dldDk_nsv_test[testv,:,:] 
    dldDk_nsv_expected = get_dldDk2D(k, nlevels, nraneffs, ZtZ[0,:,:], Zte[testv,:,:], sigma2[testv], DinvIplusZtZD[testv,:,:])[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(dldDk_nsv_test,dldDk_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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
# The below function tests the function `flattenZtZ`. It does this by 
# simulating random test data and testing the reshape was performed correctly.
#
# =============================================================================
def test_flattenZtZ():

    # Setup variables
    nraneffs = np.array([np.random.randint(2,10)])
    nlevels = np.array([30])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, n=100, nlevels=nlevels, nraneffs=nraneffs)

    # ZtZ
    ZtZ = Z.transpose() @ Z

    # Spatially varying ZtZ
    ZtZ_sv = Z_sv.transpose(0,2,1) @ Z_sv

    # q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Boolean testing whether flattening worked
    passed_sv = True
    passed = True

    # Loop through and check all expected blocks are equal
    for i in np.arange(l0):

        passed = passed and np.allclose(ZtZ_flattened[:,i*q0:(i+1)*q0],ZtZ[i*q0:(i+1)*q0,i*q0:(i+1)*q0])
        passed_sv = passed_sv and np.allclose(ZtZ_sv_flattened[testv,:,i*q0:(i+1)*q0],ZtZ_sv[testv,i*q0:(i+1)*q0,i*q0:(i+1)*q0])

    # Result
    if passed:
        result = 'Passed'
    else:
        result = 'Failed'

    # Result spatially varying
    if passed_sv:
        result_sv = 'Passed'
    else:
        result_sv = 'Failed'

    # Combine results
    result = result and result_sv
        
    print('=============================================================')
    print('Unit test for: get_dldDk3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_covdldbeta3D`. It does this by
# simulating random test data and testing against it's 2D counterpart from
# npMatrix2d.py.
#
# =============================================================================
def test_get_covdldbeta3D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # First test spatially varying
    covdldB_sv_test = get_covdldbeta3D(XtZ_sv, XtX_sv, ZtZ_diag_sv, DinvIplusZtZD_diag_sv, sigma2, nraneffs)[testv,:,:]
    covdldB_sv_expected = get_covdldbeta2D(XtZ_sv[testv,:,:], XtX_sv[testv,:,:], ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], sigma2[testv])

    # Check if results are all close.
    sv_testVal = np.allclose(covdldB_sv_test,covdldB_sv_expected)

    # Now test non spatially varying
    covdldB_nsv_test = get_covdldbeta3D(XtZ, XtX, ZtZ_diag, DinvIplusZtZD_diag, sigma2, nraneffs)[testv,:,:]
    covdldB_nsv_expected = get_covdldbeta2D(XtZ[0,:,:], XtX[0,:,:], ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], sigma2[testv])

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldB_nsv_test,covdldB_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Obtain D(I+Z'ZD)^(-1) flattened
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 
    DinvIplusZtZD_sv_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # First test spatially varying
    covdldB_sv_test = get_covdldbeta3D(XtZ_sv, XtX_sv, ZtZ_sv_flattened, DinvIplusZtZD_sv_flattened, sigma2, nraneffs)[testv,:,:]
    covdldB_sv_expected = get_covdldbeta2D(XtZ_sv[testv,:,:], XtX_sv[testv,:,:], ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], sigma2[testv])

    # Check if results are all close.
    sv_testVal = np.allclose(covdldB_sv_test,covdldB_sv_expected)

    # Now test non spatially varying
    covdldB_nsv_test = get_covdldbeta3D(XtZ, XtX, ZtZ_flattened, DinvIplusZtZD_flattened, sigma2, nraneffs)[testv,:,:]
    covdldB_nsv_expected = get_covdldbeta2D(XtZ[0,:,:], XtX[0,:,:], ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], sigma2[testv])

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldB_nsv_test,covdldB_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # First test spatially varying
    covdldB_sv_test = get_covdldbeta3D(XtZ_sv, XtX_sv, ZtZ_sv, DinvIplusZtZD_sv, sigma2, nraneffs)[testv,:,:]
    covdldB_sv_expected = get_covdldbeta2D(XtZ_sv[testv,:,:], XtX_sv[testv,:,:], ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], sigma2[testv])

    # Check if results are all close.
    sv_testVal = np.allclose(covdldB_sv_test,covdldB_sv_expected)

    # Now test non spatially varying
    covdldB_nsv_test = get_covdldbeta3D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2, nraneffs)[testv,:,:]
    covdldB_nsv_expected = get_covdldbeta2D(XtZ[0,:,:], XtX[0,:,:], ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], sigma2[testv])

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldB_nsv_test,covdldB_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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
# npMatrix2d.py.
#
# =============================================================================
def test_get_covdldDkdsigma23D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # First test spatially varying
    covdldDsigma2_sv_test,ZtZmat = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_diag_sv, DinvIplusZtZD_diag_sv, dupMatTdict)
    covdldDsigma2_sv_test,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_diag_sv, DinvIplusZtZD_diag_sv, dupMatTdict, ZtZmat=ZtZmat)
    covdldDsigma2_sv_test = covdldDsigma2_sv_test[testv,:,:]
    covdldDsigma2_sv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], dupMatTdict)

    # Check if results are all close.
    sv_testVal = np.allclose(covdldDsigma2_sv_test,covdldDsigma2_sv_expected)

    # Now test non spatially varying
    covdldDsigma2_nsv_test,ZtZmat = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_diag, DinvIplusZtZD_diag, dupMatTdict)
    covdldDsigma2_nsv_test,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_diag, DinvIplusZtZD_diag, dupMatTdict, ZtZmat=ZtZmat)
    covdldDsigma2_nsv_test = covdldDsigma2_nsv_test[testv,:,:]
    covdldDsigma2_nsv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldDsigma2_nsv_test,covdldDsigma2_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Factor number
    k = 0

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Obtain D(I+Z'ZD)^(-1) flattened
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 
    DinvIplusZtZD_sv_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # First test spatially varying
    covdldDsigma2_sv_test,ZtZmat = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_sv_flattened, DinvIplusZtZD_sv_flattened, dupMatTdict)
    covdldDsigma2_sv_test,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_sv_flattened, DinvIplusZtZD_sv_flattened, dupMatTdict, ZtZmat=ZtZmat)
    covdldDsigma2_sv_test = covdldDsigma2_sv_test[testv,:,:]
    covdldDsigma2_sv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], dupMatTdict)

    # Check if results are all close.
    sv_testVal = np.allclose(covdldDsigma2_sv_test,covdldDsigma2_sv_expected)

    # Now test non spatially varying
    covdldDsigma2_nsv_test,ZtZmat = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_flattened, DinvIplusZtZD_flattened, dupMatTdict)
    covdldDsigma2_nsv_test,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_flattened, DinvIplusZtZD_flattened, dupMatTdict, ZtZmat=ZtZmat)
    covdldDsigma2_nsv_test = covdldDsigma2_nsv_test[testv,:,:]
    covdldDsigma2_nsv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldDsigma2_nsv_test,covdldDsigma2_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # First test spatially varying
    covdldDsigma2_sv_test,ZtZmat = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_sv, DinvIplusZtZD_sv, dupMatTdict)
    covdldDsigma2_sv_test,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ_sv, DinvIplusZtZD_sv, dupMatTdict, ZtZmat=ZtZmat)
    covdldDsigma2_sv_test = covdldDsigma2_sv_test[testv,:,:]
    covdldDsigma2_sv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], dupMatTdict)

    # Check if results are all close.
    sv_testVal = np.allclose(covdldDsigma2_sv_test,covdldDsigma2_sv_expected)

    # Now test non spatially varying
    covdldDsigma2_nsv_test,ZtZmat = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)
    covdldDsigma2_nsv_test,_ = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, ZtZmat=ZtZmat)
    covdldDsigma2_nsv_test = covdldDsigma2_nsv_test[testv,:,:]
    covdldDsigma2_nsv_expected,_ = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)

    # Check if results are all close.
    nsv_testVal = np.allclose(covdldDsigma2_nsv_test,covdldDsigma2_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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
# npMatrix2d.py.
#
# =============================================================================
def test_get_covdldDk1Dk23D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # Decide on 2 random factors
    k1 = np.random.randint(0,nraneffs.shape[0])
    k2 = np.random.randint(0,nraneffs.shape[0])

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # First test spatially varying
    covdldD_sv_test,perm = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_diag_sv, DinvIplusZtZD_diag_sv, dupMatTdict)
    covdldD_sv_test,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_diag_sv, DinvIplusZtZD_diag_sv, dupMatTdict,perm=perm)
    covdldD_sv_test = covdldD_sv_test[testv,:,:]
    covdldD_sv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], dupMatTdict)[0]
  
    # Check if results are all close.
    sv_testVal = np.allclose(covdldD_sv_test,covdldD_sv_expected)

    # Now test non spatially varying
    covdldD_nsv_test,perm = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_diag, DinvIplusZtZD_diag, dupMatTdict)
    covdldD_nsv_test,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_diag, DinvIplusZtZD_diag, dupMatTdict)
    covdldD_nsv_test = covdldD_nsv_test[testv,:,:]
    covdldD_nsv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(covdldD_nsv_test,covdldD_nsv_expected)

    # Check against 2D version.
    testVal_tc1 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # Obtain D(I+Z'ZD)^(-1) flattened
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 
    DinvIplusZtZD_sv_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_sv_flattened, nlevels, nraneffs) 

    # The factors
    k1 = 0
    k2 = 0

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # First test spatially varying
    covdldD_sv_test,perm = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_sv_flattened, DinvIplusZtZD_sv_flattened, dupMatTdict)
    covdldD_sv_test,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_sv_flattened, DinvIplusZtZD_sv_flattened, dupMatTdict,perm=perm)
    covdldD_sv_test = covdldD_sv_test[testv,:,:]
    covdldD_sv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], dupMatTdict)[0]
  
    # Check if results are all close.
    sv_testVal = np.allclose(covdldD_sv_test,covdldD_sv_expected)

    # Now test non spatially varying
    covdldD_nsv_test,perm = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_flattened, DinvIplusZtZD_flattened, dupMatTdict)
    covdldD_nsv_test,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_flattened, DinvIplusZtZD_flattened, dupMatTdict)
    covdldD_nsv_test = covdldD_nsv_test[testv,:,:]
    covdldD_nsv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(covdldD_nsv_test,covdldD_nsv_expected)

    # Check against 2D version.
    testVal_tc2 = nsv_testVal and sv_testVal

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Decide on 2 random factors
    k1 = np.random.randint(0,nraneffs.shape[0])
    k2 = np.random.randint(0,nraneffs.shape[0])

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).toarray().transpose()

    # First test spatially varying
    covdldD_sv_test,perm = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_sv, DinvIplusZtZD_sv, dupMatTdict)
    covdldD_sv_test,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ_sv, DinvIplusZtZD_sv, dupMatTdict,perm=perm)
    covdldD_sv_test = covdldD_sv_test[testv,:,:]
    covdldD_sv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ_sv[testv,:,:], DinvIplusZtZD_sv[testv,:,:], dupMatTdict)[0]
  
    # Check if results are all close.
    sv_testVal = np.allclose(covdldD_sv_test,covdldD_sv_expected)

    # Now test non spatially varying
    covdldD_nsv_test,perm = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)
    covdldD_nsv_test,_ = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)
    covdldD_nsv_test = covdldD_nsv_test[testv,:,:]
    covdldD_nsv_expected = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0]
    
    # Check if results are all close.
    nsv_testVal = np.allclose(covdldD_nsv_test,covdldD_nsv_expected)

    # Check against 2D version.
    testVal_tc3 = nsv_testVal and sv_testVal

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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
# The below function tests the function `getConvergedIndices`. It does this by
# testing against a predefined example.
#
# =============================================================================
def test_getConvergedIndices():

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
    print('Unit test for: getConvergedIndices')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `block2stacked3D`. It does this by
# generating a random example and testing against it's 2D counterpart from
# npMatrix2d.py.
#
# =============================================================================
def test_block2stacked3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,60)
    n1 = np.random.randint(10,20)
    n2 = np.random.randint(10,20)
    l1 = np.random.randint(50,60)
    l2 = np.random.randint(50,60)

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
# npMatrix2d.py.
#
# =============================================================================
def test_mat2vecb3D():

    # Generate random matrix dimensions
    v = np.random.randint(10,30)
    n1 = np.random.randint(10,20)
    n2 = np.random.randint(10,20)
    l1 = np.random.randint(10,50)
    l2 = np.random.randint(10,50)

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
# npMatrix2d.py.
#
# =============================================================================
def test_sumAijBijt3D():

    # Generate random matrix dimensions
    v = np.random.randint(10,20)
    n1 = np.random.randint(2,10)
    n1prime = np.random.randint(2,10)
    n2 = np.random.randint(2,10)
    l1 = np.random.randint(50,60)
    l2 = np.random.randint(50,60)

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
# npMatrix2d.py.
#
# =============================================================================
def test_sumAijKronBij3D():

    # Generate random matrix dimensions
    v = np.random.randint(40,50)
    n1 = np.random.randint(2,10)
    n2 = np.random.randint(2,10)
    l1 = np.random.randint(50,60)
    l2 = np.random.randint(50,60)

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
# The below function tests the function `get_resms3D`. It does this by simulating
# random test data and testing against niave computation using `ssr2D` from 
# npMatrix2d.py.
#
# =============================================================================
def test_get_resms3D():

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D()
    v = Y.shape[0]
    n = X.shape[0]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # First test spatially varying
    resms_sv_test = get_resms3D(YtX_sv, YtY, XtX_sv, beta, n_sv, p)[testv,:,:]
    resms_sv_expected = ssr2D(YtX_sv[testv,:,:], YtY[testv,:,:], XtX_sv[testv,:,:], beta[testv,:,:])/(n_sv[testv]-p)

    # Check if results are all close.
    sv_testVal = np.allclose(resms_sv_test,resms_sv_expected)

    # Now test non spatially varying
    resms_nsv_test = get_resms3D(YtX, YtY, XtX, beta, n, p)[testv,:,:]
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
    print('Unit test for: get_resms3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)
    
# =============================================================================
#
# The below function tests the function `sumTTt_1fac1ran3D`. It does this by
# simulating random test data and testing against niave calculation.
#
# =============================================================================
def test_sumTTt_1fac1ran3D():

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]


    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    DinvIplusZtZD_sv = D @ np.linalg.inv(np.eye(q) + ZtZ_sv @ D)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 
    DinvIplusZtZD_diag_sv = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag_sv, nlevels, nraneffs) 

    # Work out block size
    qk = nraneffs[0]
    p = np.array([qk,1])

    # ------------------------------------------------------------------
    # Non spatially varying
    # ------------------------------------------------------------------

    # Get result using niave calculation
    sumTT_expected = sumAijBijt3D(ZtZ @ DinvIplusZtZD, ZtZ, p, p)

    # Get result using function
    sumTT_test = sumTTt_1fac1ran3D(ZtZ_diag, DinvIplusZtZD_diag, q, 1)

    # Work out the test value for non-spatially varying
    testVal_nsv = np.allclose(sumTT_test, sumTT_expected)

    # ------------------------------------------------------------------
    # Spatially varying
    # ------------------------------------------------------------------

    # Get result using niave calculation
    sumTT_expected = sumAijBijt3D(ZtZ_sv @ DinvIplusZtZD_sv, ZtZ_sv, p, p)

    # Get result using function
    sumTT_test = sumTTt_1fac1ran3D(ZtZ_diag_sv, DinvIplusZtZD_diag_sv, q, 1)

    # Work out the test value for spatially varying
    testVal_sv = np.allclose(sumTT_test, sumTT_expected)

    # Overall test value
    testVal = testVal_sv and testVal_nsv

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: sumTTt_1fac1ran3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


#sumTTt_1fac1ran3D(ZtZ, DinvIplusZtZD, l0, q0)

# =============================================================================
#
# The below function tests the function `covB3D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_get_covB3D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    q0 = nraneffs[0]
    l0 = nlevels[0]
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # Get diagonal ZtZ
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    # Get XtiVX and ZtiVX
    # ----------------------------------------------------------------------

    # Multiply by Z'X
    DinvIplusZtZDZtX = np.einsum('ij,ijk->ijk', DinvIplusZtZD_diag, ZtX)

    # Get Z'V^{-1}X
    ZtiVX = ZtX - np.einsum('ij,ijk->ijk', ZtZ_diag, DinvIplusZtZDZtX)

    # Reshape appropriately
    DinvIplusZtZDZtX = DinvIplusZtZDZtX.reshape(v,q0*l0,p)

    # Work out X'V^(-1)X and X'V^(-1)Y by dimension reduction formulae
    XtiVX = XtX - DinvIplusZtZDZtX.transpose((0,2,1)) @ ZtX

    # Run F test
    covB_test = get_covB3D(XtiVX, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc1 = np.allclose(covB_test,covB_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    q0 = nraneffs[0]
    l0 = nlevels[0]
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Flattened ZtZ
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    covB_expected = np.linalg.inv(X.transpose() @ invV @ X/sigma2[testv])

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, None, ZtZ_flattened, nlevels, nraneffs) 

    # Run F test
    covB_test = get_covB3D(XtiVX, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc2 = np.allclose(covB_test,covB_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    covB_expected = np.linalg.inv(X.transpose() @ invV @ X/sigma2[testv])

    # Using function
    # ----------------------------------------------------------------------
    
    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    covB_test = get_covB3D(XtiVX, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc3 = np.allclose(covB_test,covB_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_covB3D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `varLB3D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_get_varLB3D():

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
 
    # Get diagonal ZtZ
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    varLB_test = get_varLB3D(L, XtiVX, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc1 = np.allclose(varLB_test,varLB_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    varLB_test = get_varLB3D(L, XtiVX, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc2 = np.allclose(varLB_test,varLB_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    varLB_test = get_varLB3D(L, XtiVX, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc3 = np.allclose(varLB_test,varLB_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # Get diagonal ZtZ
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    T_test = get_T3D(L, XtiVX, beta, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc1 = np.allclose(T_test,T_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    T_test = get_T3D(L, XtiVX, beta, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc2 = np.allclose(T_test,T_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    T_test = get_T3D(L, XtiVX, beta, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc3 = np.allclose(T_test,T_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # Get diagonal ZtZ
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    F_test = get_F3D(L, XtiVX, beta, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc1 = np.allclose(F_test,F_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    F_test = get_F3D(L, XtiVX, beta, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc2 = np.allclose(F_test,F_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Run F test
    F_test = get_F3D(L, XtiVX, beta, sigma2, nraneffs)[testv,:]

    # Check if results are all close.
    testVal_tc3 = np.allclose(F_test,F_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Get D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Get diagonal ZtZ
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # X'V^(-1)X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB3D(L, XtiVX, sigma2, nraneffs)[testv,:,:]
    
    # Get derivative of S^2
    dS2 = get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2)[testv,:,:]

    # Get Fisher information matrix
    InfoMat = get_InfoMat3D(DinvIplusZtZD_diag, sigma2, n, nlevels, nraneffs, ZtZ_diag)[testv,:,:]

    # Calculate df estimator
    swdf_expected = 2*(S2**2)/(dS2.transpose() @ np.linalg.inv(InfoMat) @ dS2)

    # Calculate using function
    swdf_test = get_swdf_T3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ_diag, DinvIplusZtZD_diag, n, nlevels, nraneffs)[testv,:,:]

    # Check if results are all close.
    testVal_tc1 = np.allclose(swdf_test,swdf_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    q0 = nraneffs[0]
    l0 = nlevels[0]
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Use function to get flattened matrices
    ZtZ = flattenZtZ(ZtZ, l0, q0)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Get D(I+Z'ZD)^(-1)
    DinvIplusZtZD = get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs)

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # X'V^(-1)X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB3D(L, XtiVX, sigma2, nraneffs)[testv,:,:]
    
    # Get derivative of S^2
    dS2 = get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2)[testv,:,:]

    # Get Fisher information matrix
    InfoMat = get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ)[testv,:,:]

    # Calculate df estimator
    swdf_expected = 2*(S2**2)/(dS2.transpose() @ np.linalg.inv(InfoMat) @ dS2)

    # Calculate using function
    swdf_test = get_swdf_T3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ, DinvIplusZtZD, n, nlevels, nraneffs)[testv,:,:]

    # Check if results are all close.
    testVal_tc2 = np.allclose(swdf_test,swdf_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Get D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # X'V^(-1)X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB3D(L, XtiVX, sigma2, nraneffs)[testv,:,:]
    
    # Get derivative of S^2
    dS2 = get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2)[testv,:,:]

    # Get Fisher information matrix
    InfoMat = get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ)[testv,:,:]

    # Calculate df estimator
    swdf_expected = 2*(S2**2)/(dS2.transpose() @ np.linalg.inv(InfoMat) @ dS2)

    # Calculate using function
    swdf_test = get_swdf_T3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ, DinvIplusZtZD, n, nlevels, nraneffs)[testv,:,:]

    # Check if results are all close.
    testVal_tc3 = np.allclose(swdf_test,swdf_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Get diagonal ZtZ 
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Get diagonal ZtZ 
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # L is rL in rank
    rL = np.linalg.matrix_rank(L)

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # X'V^(-1)X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Initialize empty sum.
    sum_swdf_adj = 0

    # Loop through first rL rows of L
    for i in np.arange(rL):

        # Work out the swdf for each row of L
        swdf_row = get_swdf_T3D(L[i:(i+1),:], sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ_diag, DinvIplusZtZD_diag, n, nlevels, nraneffs)[testv,:,:]

        # Work out adjusted df = df/(df-2)
        swdf_adj = swdf_row/(swdf_row-2)

        # Add to running sum
        sum_swdf_adj = sum_swdf_adj + swdf_adj[0]

    # Work out final df
    swdf_expected = 2*sum_swdf_adj/(sum_swdf_adj-rL)

    # Function version 
    swdf_test = get_swdf_F3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ_diag, DinvIplusZtZD_diag, n, nlevels, nraneffs)[testv]

    # Check if results are all close.
    testVal_tc1 = np.allclose(swdf_test,swdf_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Flattened ZtZ
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Test result (nsv)
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # L is rL in rank
    rL = np.linalg.matrix_rank(L)

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # X'V^(-1)X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Initialize empty sum.
    sum_swdf_adj = 0

    # Loop through first rL rows of L
    for i in np.arange(rL):

        # Work out the swdf for each row of L
        swdf_row = get_swdf_T3D(L[i:(i+1),:], sigma2,  XtiVX, ZtiVX, XtZ, ZtX, ZtZ_flattened, DinvIplusZtZD_flattened, n, nlevels, nraneffs)[testv,:,:]

        # Work out adjusted df = df/(df-2)
        swdf_adj = swdf_row/(swdf_row-2)

        # Add to running sum
        sum_swdf_adj = sum_swdf_adj + swdf_adj[0]

    # Work out final df
    swdf_expected = 2*sum_swdf_adj/(sum_swdf_adj-rL)

    # Function version 
    swdf_test = get_swdf_F3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ_flattened, DinvIplusZtZD_flattened, n, nlevels, nraneffs)[testv]

    # Check if results are all close.
    testVal_tc2 = np.allclose(swdf_test,swdf_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # L is rL in rank
    rL = np.linalg.matrix_rank(L)

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # X'V^(-1)X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Initialize empty sum.
    sum_swdf_adj = 0

    # Loop through first rL rows of L
    for i in np.arange(rL):

        # Work out the swdf for each row of L
        swdf_row = get_swdf_T3D(L[i:(i+1),:], sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ, DinvIplusZtZD, n, nlevels, nraneffs)[testv,:,:]

        # Work out adjusted df = df/(df-2)
        swdf_adj = swdf_row/(swdf_row-2)

        # Add to running sum
        sum_swdf_adj = sum_swdf_adj + swdf_adj[0]

    # Work out final df
    swdf_expected = 2*sum_swdf_adj/(sum_swdf_adj-rL)

    # Function version 
    swdf_test = get_swdf_F3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ, DinvIplusZtZD, n, nlevels, nraneffs)[testv]

    # Check if results are all close.
    testVal_tc3 = np.allclose(swdf_test,swdf_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Niave calculation for one voxel.
    # ----------------------------------------------------------------------------------
    IplusZDZt = np.eye(n) + Z @ D[testv,:,:] @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    # X'V^(-1)X
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # X'V^(-1)X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2_expected = np.zeros((1+np.int32(np.sum(nraneffs*(nraneffs+1)/2)),1))

    # Work out indices for each start of each component of vector 
    # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
    DerivInds = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    DerivInds = np.insert(DerivInds,0,1)

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.inv(XtiVX[testv,:,:]) @ L.transpose()

    # Add to dS2
    dS2_expected[0:1] = dS2dsigma2.reshape(dS2_expected[0:1].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((np.int32(nraneffs[k]*(nraneffs[k]+1)/2),1))

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
                    
            # Work out Z_(k,j)'
            Zkjt = Z[:,Ikj].transpose()

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = Zkjt @ invV @ X

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.inv(XtiVX[testv,:,:]) @ L.transpose()
            
            # Sum terms
            dS2dvechDk = dS2dvechDk + dupMat2D(nraneffs[k]).toarray().transpose() @ mat2vec2D(np.kron(K,K.transpose()))

        # Multiply by sigma^2
        dS2dvechDk = sigma2[testv]*dS2dvechDk

        # Add to dS2
        dS2_expected[DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2_expected[DerivInds[k]:DerivInds[k+1]].shape)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Get diagonal ZtZ
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    # Obtain result from function
    dS2_test = get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2)[testv,:,:]

    # Check if results are all close.
    testVal_tc1 = np.allclose(dS2_test,dS2_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Calculate Z'V^{-1}X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2_expected = np.zeros((1+np.int32(np.sum(nraneffs*(nraneffs+1)/2)),1))

    # Work out indices for each start of each component of vector 
    # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
    DerivInds = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    DerivInds = np.insert(DerivInds,0,1)

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.inv(XtiVX[testv,:,:]) @ L.transpose()

    # Add to dS2
    dS2_expected[0:1] = dS2dsigma2.reshape(dS2_expected[0:1].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((np.int32(nraneffs[k]*(nraneffs[k]+1)/2),1))

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
                    
            # Work out Z_(k,j)'
            Zkjt = Z[:,Ikj].transpose()

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = Zkjt @ invV @ X

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.inv(XtiVX[testv,:,:]) @ L.transpose()
            
            # Sum terms
            dS2dvechDk = dS2dvechDk + dupMat2D(nraneffs[k]).toarray().transpose() @ mat2vec2D(np.kron(K,K.transpose()))

        # Multiply by sigma^2
        dS2dvechDk = sigma2[testv]*dS2dvechDk

        # Add to dS2
        dS2_expected[DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2_expected[DerivInds[k]:DerivInds[k+1]].shape)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Flattened ZtZ
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)

    # Test result (nsv)
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, D, ZtZ_flattened, nlevels, nraneffs) 

    # Obtain result from function
    dS2_test = get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2)[testv,:,:]

    # Check if results are all close.
    testVal_tc2 = np.allclose(dS2_test,dS2_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

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
    XtiVX = X.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # Calculate Z'V^{-1}X
    ZtiVX = Z.transpose() @ np.linalg.inv(np.eye(n) + Z @ D @ Z.transpose()) @ X

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2_expected = np.zeros((1+np.int32(np.sum(nraneffs*(nraneffs+1)/2)),1))

    # Work out indices for each start of each component of vector 
    # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
    DerivInds = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    DerivInds = np.insert(DerivInds,0,1)

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.inv(XtiVX[testv,:,:]) @ L.transpose()

    # Add to dS2
    dS2_expected[0:1] = dS2dsigma2.reshape(dS2_expected[0:1].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((np.int32(nraneffs[k]*(nraneffs[k]+1)/2),1))

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
                    
            # Work out Z_(k,j)'
            Zkjt = Z[:,Ikj].transpose()

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = Zkjt @ invV @ X

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.inv(XtiVX[testv,:,:]) @ L.transpose()
            
            # Sum terms
            dS2dvechDk = dS2dvechDk + dupMat2D(nraneffs[k]).toarray().transpose() @ mat2vec2D(np.kron(K,K.transpose()))

        # Multiply by sigma^2
        dS2dvechDk = sigma2[testv]*dS2dvechDk

        # Add to dS2
        dS2_expected[DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2_expected[DerivInds[k]:DerivInds[k+1]].shape)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Obtain result from function
    dS2_test = get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2)[testv,:,:]

    # Check if results are all close.
    testVal_tc3 = np.allclose(dS2_test,dS2_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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

    # -------------------------------------------------------------------------
    # Test case 1: 1 random factor, 1 random effect
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([1])
    nlevels = np.array([800])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = X.shape[0]

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()

    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    FishIndsDk = np.insert(FishIndsDk,0,1)

    # Initialize FIsher Information matrix
    FI_expected = np.zeros((tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2[testv]**2))
    
    # Add dl/dsigma2 covariance
    FI_expected[0,0] = covdldsigma2

    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nraneffs)):

        # Get covariance of dldsigma and dldD      
        covdldsigmadD = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0].reshape(FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FI_expected[0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
        FI_expected[FishIndsDk[k]:FishIndsDk[k+1],0:1] = FI_expected[0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose()
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nraneffs)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0]

            # Add to FImat
            FI_expected[np.ix_(IndsDk1, IndsDk2)] = covdldDk1dDk2
            FI_expected[np.ix_(IndsDk2, IndsDk1)] = FI_expected[np.ix_(IndsDk1, IndsDk2)].transpose()

    # Get diagonal ZtZ
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_diag = get_DinvIplusZtZD3D(Ddict, None, ZtZ_diag, nlevels, nraneffs) 

    FI_test = get_InfoMat3D(DinvIplusZtZD_diag, sigma2, n, nlevels, nraneffs, ZtZ_diag)[testv,:,:]

    # Check if results are all close.
    testVal_tc1 = np.allclose(FI_test,FI_expected)

    # -------------------------------------------------------------------------
    # Test case 2: 1 random factor, multiple random effects
    # -------------------------------------------------------------------------

    # Setup variables
    nraneffs = np.array([2])
    nlevels = np.array([300])
    q = np.sum(nlevels*nraneffs)
    v = 10

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)
    n = Y.shape[1]
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = X.shape[0]

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Flattened ZtZ
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)

    # Work out the indices in D where a new block Dk appears
    Dinds = np.cumsum(nlevels*nraneffs)
    Dinds = np.insert(Dinds,0,0)

    # New empty D dict
    Ddict = dict()

    # Work out Dk for each factor, factor k 
    for k in np.arange(nlevels.shape[0]):

        # Add Dk to the dict
        Ddict[k] = D[:,Dinds[k]:(Dinds[k]+nraneffs[k]),Dinds[k]:(Dinds[k]+nraneffs[k])]

    # Obtain D(I+Z'ZD)^(-1) diag
    DinvIplusZtZD_flattened = get_DinvIplusZtZD3D(Ddict, None, ZtZ_flattened, nlevels, nraneffs) 

    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()

    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    FishIndsDk = np.insert(FishIndsDk,0,1)

    # Initialize FIsher Information matrix
    FI_expected = np.zeros((tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2[testv]**2))
    
    # Add dl/dsigma2 covariance
    FI_expected[0,0] = covdldsigma2

    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nraneffs)):

        # Get covariance of dldsigma and dldD      
        covdldsigmadD = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0].reshape(FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FI_expected[0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
        FI_expected[FishIndsDk[k]:FishIndsDk[k+1],0:1] = FI_expected[0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose()
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nraneffs)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0]

            # Add to FImat
            FI_expected[np.ix_(IndsDk1, IndsDk2)] = covdldDk1dDk2
            FI_expected[np.ix_(IndsDk2, IndsDk1)] = FI_expected[np.ix_(IndsDk1, IndsDk2)].transpose()

    FI_test = get_InfoMat3D(DinvIplusZtZD_flattened, sigma2, n, nlevels, nraneffs, ZtZ_flattened)[testv,:,:]

    # Check if results are all close.
    testVal_tc2 = np.allclose(FI_test,FI_expected)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    n = Y.shape[1]
    q = np.sum(nlevels*nraneffs)
    p = X.shape[1]

    # Generate product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = X.shape[0]

    # Choose random voxel to check worked correctly
    testv = np.random.randint(0,v)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()

    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    FishIndsDk = np.insert(FishIndsDk,0,1)

    # Initialize FIsher Information matrix
    FI_expected = np.zeros((tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2[testv]**2))
    
    # Add dl/dsigma2 covariance
    FI_expected[0,0] = covdldsigma2

    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nraneffs)):

        # Get covariance of dldsigma and dldD      
        covdldsigmadD = get_covdldDkdsigma22D(k, sigma2[testv], nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0].reshape(FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FI_expected[0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
        FI_expected[FishIndsDk[k]:FishIndsDk[k+1],0:1] = FI_expected[0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose()
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nraneffs)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ[0,:,:], DinvIplusZtZD[testv,:,:], dupMatTdict)[0]

            # Add to FImat
            FI_expected[np.ix_(IndsDk1, IndsDk2)] = covdldDk1dDk2
            FI_expected[np.ix_(IndsDk2, IndsDk1)] = FI_expected[np.ix_(IndsDk1, IndsDk2)].transpose()

    FI_test = get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ)[testv,:,:]

    # Check if results are all close.
    testVal_tc3 = np.allclose(FI_test,FI_expected)

    # Combine the test values
    testVal = testVal_tc3 and testVal_tc2 and testVal_tc1

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

    # Test get_DinvIplusZtZD3D
    name = 'get_DinvIplusZtZD3D'
    result = test_get_DinvIplusZtZD3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

    # Test flattenZtZ
    name = 'flattenZtZ'
    result = test_flattenZtZ()
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


    # Test getConvergedIndices
    name = 'getConvergedIndices'
    result = test_getConvergedIndices()
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


    # Test sumTTt_1fac1ran3D
    name = 'sumTTt_1fac1ran3D'
    result = test_sumTTt_1fac1ran3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_resms3D
    name = 'get_resms3D'
    result = test_get_resms3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

        
    # Test get_covB3D
    name = 'get_covB3D'
    result = test_get_covB3D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_varLB3D
    name = 'get_varLB3D'
    result = test_get_varLB3D()
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
