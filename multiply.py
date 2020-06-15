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
from scipy import stats

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.npMatrix2d import *
from lib.npMatrix3d import *
from test.Unit.genTestDat import prodMats3D, genTestData3D

# Setup variables
nraneffs = np.array([1])
nlevels = np.array([1500])
q = np.sum(nlevels*nraneffs)
v = 13

# Generate test data
Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D,X_sv,Z_sv,n_sv = genTestData3D(v=v, nlevels=nlevels, nraneffs=nraneffs)

# Generate product matrices
XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, XtX_sv, XtY_sv, XtZ_sv, YtX_sv, YtZ_sv, ZtX_sv, ZtY_sv, ZtZ_sv = prodMats3D(Y,Z,X,Z_sv,X_sv)

ZtZ = ZtZ_sv

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
DinvIplusZtZD = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

# First random factor
k=0

# Get the indices for the factors 
Ik = fac_indices2D(k, nlevels, nraneffs)

# Work out lk
lk = nlevels[k]

# Work out block size
qk = nraneffs[k]
p = np.array([qk,1])

# Zte
Zte = ZtY - ZtX @ beta

t1 = time.time()
# Initalize D to zeros
invSig2ZteetZminusZtZ = np.zeros((Zte.shape[0],nraneffs[k],nraneffs[k]))

# First we work out the derivative we require.
for j in np.arange(nlevels[k]):

    Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

    # Work out Z_(k, j)'Z_(k, j)
    ZkjtZkj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)]

    # Work out Z_(k,j)'e
    Zkjte = Zte[:, Ikj,:]

    if j==0:

        # Add first \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
        invSig2ZteetZminusZtZ = np.einsum('i,ijk->ijk',1/sigma2,(Zkjte @ Zkjte.transpose(0,2,1))) - ZkjtZkj

    else:

        # Add next \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
        invSig2ZteetZminusZtZ = invSig2ZteetZminusZtZ + np.einsum('i,ijk->ijk',1/sigma2,(Zkjte @ Zkjte.transpose(0,2,1))) - ZkjtZkj
t2 = time.time()
print('new time: ', t2-t1)


t1 = time.time()
diagZtZ = np.einsum('ijj->ij', ZtZ)

# Work out block size
qk = nraneffs[k]
p = np.array([qk,1])

# second version
invSig2ZteetZminusZtZ2 = np.einsum('i,ijk->ijk',1/sigma2,sumAijBijt3D(Zte, Zte, p, p)) - diagZtZ
t2 = time.time()
print('new time: ', t2-t1)