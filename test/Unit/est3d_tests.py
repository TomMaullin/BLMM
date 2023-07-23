import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from blmm.src.npMatrix2d import mat2vech2D
from blmm.src.est2d import *
from blmm.src.est3d import *
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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = FS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_diag_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = FS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: FS')
    print('--------------------------------------------------------------')
    print('   Test case: 1 random effect, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = FS3D(XtX, XtY, ZtX, ZtY, ZtZ_diag, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = FS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = FS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv_flattened, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = FS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: FS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = FS3D(XtX, XtY, ZtX, ZtY, ZtZ_flattened, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = FS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    q = np.sum(nlevels*nraneffs)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = FS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = FS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: FS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, Multiple random factors')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = FS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = FS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

# =============================================================================
#
# The below function tests the function `pFS3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from est2d.py.
#
# =============================================================================
def test_pFS3D():

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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = pFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_diag_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = pFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: pFS')
    print('--------------------------------------------------------------')
    print('   Test case: 1 random effect, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = pFS3D(XtX, XtY, ZtX, ZtY, ZtZ_diag, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = pFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = pFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv_flattened, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = pFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: pFS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = pFS3D(XtX, XtY, ZtX, ZtY, ZtZ_flattened, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = pFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    q = np.sum(nlevels*nraneffs)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = pFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = pFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: pFS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, Multiple random factors')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = pFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = pFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

# =============================================================================
#
# The below function tests the function `SFS3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from est2d.py.
#
# =============================================================================
def test_SFS3D():

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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = SFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_diag_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = SFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: SFS')
    print('--------------------------------------------------------------')
    print('   Test case: 1 random effect, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = SFS3D(XtX, XtY, ZtX, ZtY, ZtZ_diag, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = SFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = SFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv_flattened, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = SFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: SFS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = SFS3D(XtX, XtY, ZtX, ZtY, ZtZ_flattened, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = SFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    q = np.sum(nlevels*nraneffs)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    paramVec3D_sv = SFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]

    # Spatially varying 2D
    paramVec2D_sv = SFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: SFS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, Multiple random factors')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())

    # Non-spatially varying 3D
    paramVec3D_nsv = SFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]

    # Non-spatially varying 2D
    paramVec2D_nsv = SFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())

# =============================================================================
#
# The below function tests the function `pSFS3D`. It does this by simulating
# random test data and testing against it's 2D counterpart from est2d.py.
#
# =============================================================================
def test_pSFS3D():

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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Get diagonal ZtZ and ZtZ_sv
    ZtZ_diag = np.einsum('ijj->ij', ZtZ)
    ZtZ_diag_sv = np.einsum('ijj->ij', ZtZ_sv)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    t1 = time.time()
    paramVec3D_sv = pSFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_diag_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]
    t2 = time.time()

    # Spatially varying 2D
    paramVec2D_sv = pSFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: pSFS')
    print('--------------------------------------------------------------')
    print('   Test case: 1 random effect, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())
    print('      Computation time: ', t2-t1)


    # Non-spatially varying 3D
    t1 = time.time()
    paramVec3D_nsv = pSFS3D(XtX, XtY, ZtX, ZtY, ZtZ_diag, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]
    t2 = time.time()

    # Non-spatially varying 2D
    paramVec2D_nsv = pSFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())
    print('      Computation time: ', t2-t1)

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
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # Use function to get flattened matrices
    ZtZ_flattened = flattenZtZ(ZtZ, l0, q0)
    ZtZ_sv_flattened = flattenZtZ(ZtZ_sv, l0, q0)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    t1 = time.time()
    paramVec3D_sv = pSFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv_flattened, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]
    t2 = time.time()

    # Spatially varying 2D
    paramVec2D_sv = pSFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: pSFS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, 1 random factor')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())
    print('      Computation time: ', t2-t1)

    # Non-spatially varying 3D
    t1 = time.time()
    paramVec3D_nsv = pSFS3D(XtX, XtY, ZtX, ZtY, ZtZ_flattened, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]
    t2 = time.time()

    # Non-spatially varying 2D
    paramVec2D_nsv = pSFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())
    print('      Computation time: ', t2-t1)

    # -------------------------------------------------------------------------
    # Test case 3: multiple random factors, multiple random effects
    # -------------------------------------------------------------------------

    # Generate a random mass univariate linear mixed model.
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v)
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)
    q = np.sum(nlevels*nraneffs)
    n = Y.shape[1]
    p = X.shape[1]
    n_sv = n_sv.reshape(n_sv.shape[0])

    # Choose random voxel to check worked correctly
    v = Y.shape[0]
    testv = np.random.randint(0,v)

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------
    paramVec_true = beta[testv,:]
    paramVec_true = np.concatenate((paramVec_true,sigma2[testv].reshape(1,1)),axis=0)

    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[testv,facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2[testv]
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # ------------------------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------------------------
    # Spatially varying 3D
    t1 = time.time()
    paramVec3D_sv = pSFS3D(XtX_sv, XtY_sv, ZtX_sv, ZtY_sv, ZtZ_sv, XtZ_sv, YtZ_sv, YtY, YtX_sv, nlevels, nraneffs, 1e-6,n_sv)[testv,:,:]
    t2 = time.time()

    # Spatially varying 2D
    paramVec2D_sv = pSFS2D(XtX_sv[testv,:,:], XtY_sv[testv,:,:], ZtX_sv[testv,:,:], ZtY_sv[testv,:,:], ZtZ_sv[testv,:,:], XtZ_sv[testv,:,:], YtZ_sv[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n_sv[testv])[0]


    print('==============================================================')
    print('Unit test for: pSFS')
    print('--------------------------------------------------------------')
    print('   Test case: Multiple random effects, Multiple random factors')
    print('--------------------------------------------------------------')
    print('      Results (Spatially Varying): ')
    print('         ')
    print('         Truth: ', paramVec_true.transpose())
    print('         3D:    ', paramVec3D_sv.transpose())
    print('         2D:    ', paramVec2D_sv.transpose())
    print('      Computation time: ', t2-t1)

    # Non-spatially varying 3D
    t1 = time.time()
    paramVec3D_nsv = pSFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, 1e-6,n)[testv,:,:]
    t2 = time.time()

    # Non-spatially varying 2D
    paramVec2D_nsv = pSFS2D(XtX[0,:,:], XtY[testv,:,:], ZtX[0,:,:], ZtY[testv,:,:], ZtZ[0,:,:], XtZ[0,:,:], YtZ[testv,:,:], YtY[testv,:,:], YtX_sv[testv,:,:], nlevels, nraneffs, 1e-6,n)[0]

    print('--------------------------------------------------------------')
    print('      Results (Non Spatially Varying): ')
    print('         ')
    print('         3D:    ', paramVec3D_nsv.transpose())
    print('         2D:    ', paramVec2D_nsv.transpose())
    print('      Computation time: ', t2-t1)


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
