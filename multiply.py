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

# Assume A, B have same struct as Z'Z or D(I+Z'ZD)^(-1)
def sumTTt_1factor3D(ZtZk, DinvIplusZtZDk, l0, q0):

    # Number of voxels, v
    v = DinvIplusZtZDk.shape[0]

    # Work out the diagonal values the matrix product Z'Z_(k)D_(k)(I+Z'Z_(k)D_(k))^(-1)Z'Z_(k)
    DiagVals = np.einsum('ijj->ij', DinvIplusZtZDk)*np.einsum('ijj->ij', ZtZk)**2

    # Reshape diag vals and sum apropriately
    DiagVals = np.sum(DiagVals.reshape(v,q0,l0),axis=2)

    # Put values back into a matrix
    sumTTt = np.zeros((v,q0,q0))
    np.einsum('ijj->ij', sumTTt)[...] = DiagVals

    return(sumTTt)


# Setup variables
nraneffs = np.array([1])
nlevels = np.array([1500])
q = np.sum(nlevels*nraneffs)
v = 10

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

t1 = time.time()
# Work out the second term in TT'
#secondTerm = sumAijBijt3D(ZtZ[:,Ik,:] @ DinvIplusZtZD, ZtZ[:,Ik,:], p, p)
t2 = time.time()
print('timing1: ', t2-t1)

t1 = time.time()
# Work out the second term in TT'
#secondTerm2 = sumTTt_1factor3D(ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ik,Ik)], DinvIplusZtZD[np.ix_(np.arange(v),Ik,Ik)], nlevels[0], nraneffs[0])
t2 = time.time()
print('timing2: ', t2-t1)

#print(np.allclose(secondTerm2, secondTerm))

Rk1k2 = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ik,Ik)] - (ZtZ[:,Ik,:] @ DinvIplusZtZD @ ZtZ[:,:,Ik])

# Work out block sizes
p = np.array([nraneffs[k],nraneffs[k]])

# Obtain permutation
perm = None
t1 = time.time()
RkRSum,perm=sumAijKronBij3D(Rk1k2, Rk1k2, p, perm)
t2 = time.time()
print('original func (wo perm): ', t2-t1)

t1 = time.time()
RkRSum,perm=sumAijKronBij3D(Rk1k2, Rk1k2, p, perm)
t2 = time.time()
print('original func (w perm): ', t2-t1)

def sumRkronR_1factor(R, q0, l0):

    # Number of voxels
    v = R.shape[0]

    # Get diagonal values of R and reshape them
    DiagVals = np.einsum('ijj->ij', R).reshape(v, q0, l0)

    # Get Kron of the diagonal values and sum lk out
    kronDiagSum = np.sum(kron3D(DiagVals,DiagVals),axis=2)

    # Make zero array to hold result
    RkR = np.zeros((v, q0**2, q0**2))

    # Add result in
    np.einsum('ijj->ij', RkR)[...] = kronDiagSum

    return(RkR)

t1 = time.time()
RkRSum2 = sumRkronR_1factor(Rk1k2, nraneffs[0], nlevels[0])
t2 = time.time()
print('new: ', t2-t1)

print(np.allclose(RkRSum2,RkRSum))
