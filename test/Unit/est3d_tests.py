import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.npMatrix2d import mat2vech2D
from lib.est2d import *
from lib.est3d import *
from genTestDat import genTestData3D, prodMats3D

# =============================================================================
# 
# This file contains some rudimentary tests to verify that the 3D and 2D test
# cases are giving roughly the same results. 
#
# -----------------------------------------------------------------------------
#
# Author: Tom Maullin
# Last edited: 06/04/2020
#
# =============================================================================

# =============================================================================
#
# The below function tests the function `FS3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from est2d.py.
#
# =============================================================================
def test_FS3D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=4)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Work out q and n
    q = np.sum(nlevels*nparams)
    n = X.shape[0]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nparams*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = FS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nparams, 1e-6,n_sv)[testv,:,:]
    
    # Spatially varying 2D
    paramVec2D_sv = FS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n_sv[testv])[0]

    print('=============================================================')
    print('Unit test for: FS')
    print('-------------------------------------------------------------')
    print('Results (Spatially Varying): ')
    print('   ')
    print('   Truth: ', paramVec_true.transpose())
    print('   3D:    ', paramVec3D_sv.transpose())
    print('   2D:    ', paramVec2D_sv.transpose())

    
    # Non-spatially varying 3D
    paramVec3D_nsv = FS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = FS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n)[0]

    print('-------------------------------------------------------------')
    print('Results (Non Spatially Varying): ')
    print('   ')
    print('   3D:    ', paramVec3D_nsv.transpose())
    print('   2D:    ', paramVec2D_nsv.transpose())

# =============================================================================
#
# The below function tests the function `pFS3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from est2d.py.
#
# =============================================================================
def test_pFS3D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=4)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Work out q and n
    q = np.sum(nlevels*nparams)
    n = X.shape[0]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)
    
    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nparams*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)
        
    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = pFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nparams, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = pFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n_sv[testv])[0]

    print('=============================================================')
    print('Unit test for: pFS')
    print('-------------------------------------------------------------')
    print('Results (Spatially Varying): ')
    print('   ')
    print('   Truth: ', paramVec_true.transpose())
    print('   3D:    ', paramVec3D_sv.transpose())
    print('   2D:    ', paramVec2D_sv.transpose())
    
    # Non-spatially varying 3D
    paramVec3D_nsv = pFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = pFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n)[0]

    print('-------------------------------------------------------------')
    print('Results (Non Spatially Varying): ')
    print('   ')
    print('   3D:    ', paramVec3D_nsv.transpose())
    print('   2D:    ', paramVec2D_nsv.transpose())

# =============================================================================
#
# The below function tests the function `SFS3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from est2d.py.
#
# =============================================================================
def test_SFS3D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=4)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Work out q and n
    q = np.sum(nlevels*nparams)
    n = X.shape[0]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nparams*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = SFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nparams, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = SFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n_sv[testv])[0]

    print('=============================================================')
    print('Unit test for: SFS')
    print('-------------------------------------------------------------')
    print('Results (Spatially Varying): ')
    print('   ')
    print('   Truth: ', paramVec_true.transpose())
    print('   3D:    ', paramVec3D_sv.transpose())
    print('   2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = SFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = SFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n)[0]

    print('-------------------------------------------------------------')
    print('Results (Non Spatially Varying): ')
    print('   ')
    print('   3D:    ', paramVec3D_nsv.transpose())
    print('   2D:    ', paramVec2D_nsv.transpose())


# =============================================================================
#
# The below function tests the function `pSFS3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from est2d.py.
#
# =============================================================================
def test_pSFS3D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=4)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

    # Work out q and n
    q = np.sum(nlevels*nparams)
    n = X.shape[0]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nparams*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = pSFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nparams, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = pSFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n_sv[testv])[0]


    print('=============================================================')
    print('Unit test for: pSFS')
    print('-------------------------------------------------------------')
    print('Results (Spatially Varying): ')
    print('   ')
    print('   Truth: ', paramVec_true.transpose())
    print('   3D:    ', paramVec3D_sv.transpose())
    print('   2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = pSFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = pSFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nparams, 1e-6,n)[0]

    # Check if results are all close.
    sv_testVal = np.allclose(paramVec2D_sv,paramVec3D_sv,rtol=1e-2)
    nsv_testVal = np.allclose(paramVec2D_nsv,paramVec3D_nsv,rtol=1e-2)

    print('-------------------------------------------------------------')
    print('Results (Non Spatially Varying): ')
    print('   ')
    print('   3D:    ', paramVec3D_nsv.transpose())
    print('   2D:    ', paramVec2D_nsv.transpose())


# =============================================================================
#
# The below function runs all unit tests and outputs the results.
#
# =============================================================================
def run_all3D():

    # Test FS3D
    test_FS3D()

    # Test pFS3D
    test_pFS3D()

    # Test SFS3D
    test_SFS3D()

    # Test FS3D
    test_pSFS3D()

    print('=============================================================')

    print('Tests completed')
    print('=============================================================')
