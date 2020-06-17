import os
import sys
import numpy as np
import cvxopt
from cvxopt import matrix,spmatrix
import pandas as pd
import time
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
from test.Unit.genTestDat import genTestData2D, prodMats2D
from lib.est2d import *
from lib.npMatrix2d import *
from lib.cvxMatrix2d import *
from lib.PeLS import PeLS2D, PeLS2D_getSigma2, PeLS2D_getBeta, PeLS2D_getD

from cvxopt import cholmod

def recodeFactor(factor):

    # New copy of factor vector
    factor = np.array(factor)

    # Work out unique levels of the factor
    uniqueValues = np.unique(factor)
    
    # Loop through the levels replacing them with
    # a 0:l coding where l is the number of levels
    for i in np.arange(len(uniqueValues)):

        factor[factor==uniqueValues[i]]=i

    return(factor)

# Read in csvs
dataFull = pd.read_csv('./schools_full.csv')
dataReduced = pd.read_csv('./schools_reduced.csv')

# Number of subjects in full model
nf = len(dataFull)

# Number of subjects in reduced model
nr = len(dataReduced)

# Work out factors for full model
schlfac_full = recodeFactor(np.array(dataFull['schlid'].values))
studfac_full = recodeFactor(np.array(dataFull['studid'].values))
tchrfac_full = recodeFactor(np.array(dataFull['tchrid'].values))

# Work out math and year for full model
math_full = np.array(dataFull['math'].values).reshape(len(dataFull),1)
year_full = np.array(dataFull['year'].values).reshape(len(dataFull),1)

# Work out factors for reduced model
studfac_red = recodeFactor(np.array(dataReduced['studid'].values))
tchrfac_red = recodeFactor(np.array(dataReduced['tchrid'].values))

# Work out math and year for reduced model
math_red = np.array(dataReduced['math'].values).reshape(len(dataReduced),1)
year_red = np.array(dataReduced['year'].values).reshape(len(dataReduced),1)

# Construct X for full model
X_full = np.concatenate((np.ones((nf,1)),year_full),axis=1)
Y_full = math_full

# Construct X for reduced model
X_red = np.concatenate((np.ones((nr,1)),year_red),axis=1)
Y_red = math_red

# Construct Z for full model
Z_f1_full = np.zeros((nf,len(np.unique(schlfac_full))))
Z_f1_full[np.arange(nf),schlfac_full] = 1
Z_f2_full = np.zeros((nf,len(np.unique(studfac_full))))
Z_f2_full[np.arange(nf),studfac_full] = 1
Z_f3_full = np.zeros((nf,len(np.unique(tchrfac_full))))
Z_f3_full[np.arange(nf),tchrfac_full] = 1
Z_full = np.concatenate((Z_f1_full,Z_f2_full,Z_f3_full),axis=1)

print(Z_full.shape)

# Construct Z for reduced model
Z_f1_red = np.zeros((nr,len(np.unique(studfac_red))))
Z_f1_red[np.arange(nr),studfac_red] = 1
Z_f2_red = np.zeros((nr,len(np.unique(tchrfac_red))))
Z_f2_red[np.arange(nr),tchrfac_red] = 1
Z_red = np.concatenate((Z_f1_red,Z_f2_red),axis=1)

print(Z_red.shape)

# Convergence tolerance
tol = 1e-6

# nlevels for Full
nlevels_full = np.array([len(np.unique(schlfac_full)),len(np.unique(studfac_full)),len(np.unique(tchrfac_full))])

# nlevels for reduced
nlevels_red = np.array([len(np.unique(studfac_red)),len(np.unique(tchrfac_red))])

# nraneffs for full
nraneffs_full = np.array([1,1,1])

# nraneffs for full
nraneffs_red = np.array([1,1])

# Get the product matrices for full
#XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y_full,Z_full,X_full)

# Run Fisher Scoring
#t1 = time.time()
#paramVector_FS,_,nit,llh = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels_full, nraneffs_full, tol, nf, init_paramVector=None)
#t2 = time.time()

#print(t2-t1)
#print(paramVector_FS)

# Get the product matrices for reduced
#t1 = time.time()
#XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y_full,Z_full,X_full)

# Run Fisher Scoring
#paramVector_FS,_,nit,llh = cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels_red, nraneffs_red, tol, nr, init_paramVector=None)
#t2 = time.time()

#t1 = time.time()
#XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y_full,Z_full,X_full)

# Run Fisher Scoring
#paramVector_FS,_,nit,llh = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels_red, nraneffs_red, tol, nr, init_paramVector=None)
#t2 = time.time()

#print(t2-t1)
#print(paramVector_FS)

#t1 = time.time()
#XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y_red,Z_red,X_red)

# Run Fisher Scoring
#paramVector_FS,_,nit,llh = SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels_red, nraneffs_red, tol, nr, init_paramVector=None)
#t2 = time.time()


#print(t2-t1)
#print(paramVector_FS)

print('running')
t1 = time.time()
XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y_red,Z_red,X_red)

# Run Fisher Scoring
paramVector_FS,_,nit,llh = pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels_red, nraneffs_red, tol, nr, init_paramVector=None)
t2 = time.time()


#print(t2-t1)
print(paramVector_FS)

a = np.array([[1]])
b = np.array([[2]])

m = np.zeros((2,2))

t1 = time.time()
m[0,0]=a[:,:]
m[1,1]=b[:,:]
t2 = time.time()
print(t2-t1)

t1 = time.time()
m2 = scipy.linalg.block_diag(a,b)
t2 = time.time()
print(t2-t1)
#t1 = time.time(); tmp=recursiveInverse2D2(ZtZ, nraneffs_full, nlevels_full); t2 = time.time(); print(t2-t1)

a = np.array([[1,3,0,0,0,0],[2,4,0,0,0,0],[0,0,5,7,0,0],[0,0,6,8,0,0],[0,0,0,0,9,11],[0,0,0,0,10,12]])

want_row = np.array([0,1,0,1,2,3,2,3,4,5,4,5])

want_col = np.array([0,0,1,1,2,2,3,3,4,4,5,5])

print(a)
print(a[want_row,want_col])

q = 2
b = 3

def sparsePINV(ZtZ,k=1000):
    ZtZ = scipy.sparse.csr_matrix(ZtZ)
    u,s,v = scipy.sparse.linalg.svds(ZtZ,k)
    u[u<1e-10]=0
    v[v<1e-10]=0

    u = scipy.sparse.csr_matrix(u)
    v = scipy.sparse.csr_matrix(v)

    invZtZ = u @ scipy.sparse.diags(1/s) @ v

    return(invZtZ.toarray().transpose())

#print(np.mean(np.diag(ZtZ @ tmp - np.eye(ZtZ.shape[0]))))

t1 = time.time()
#tmp=sparsePINV(ZtZ)
t2 = time.time()
print(t2-t1)


underlying=np.repeat(np.arange(b)*q,q**2)

attempt_col = underlying + np.tile(np.repeat(np.arange(q),q), b)

attempt_row = underlying + np.tile(np.arange(q),q*b)

a[attempt_row,attempt_col].reshape(b,q,q)

a2 = np.array(a,dtype=np.float64)
#a2[attempt_row,attempt_col]=np.linalg.inv(a[attempt_row,attempt_col].reshape(b,q,q).transpose(0,2,1)).reshape(b*q*q)


print('running')
t1 = time.time()
XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y_full,Z_full,X_full)

# Run Fisher Scoring
#paramVector_FS,_,nit,llh = pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels_full, nraneffs_full, tol, nf, init_paramVector=None)
t2 = time.time()


t1 = time.time()
ZtZ_loc = np.nonzero(ZtZ)
ZtZ_row = ZtZ_loc[0]
ZtZ_col = ZtZ_loc[1]
ZtZ_dat = ZtZ[ZtZ_loc]

d_sigma2t = np.diag(np.repeat(np.array([89]), nlevels_full[0]))
d_sigma2s = np.diag(np.repeat(np.array([9.2]), nlevels_full[1]))
d_sigma2sch = np.diag(np.repeat(np.array([100]), nlevels_full[2]))

D = scipy.linalg.block_diag(d_sigma2t,d_sigma2s,d_sigma2sch)


D_loc = np.nonzero(D)
D_row = D_loc[0]
D_col = D_loc[1]
D_dat = D[D_loc]

D_sparse = cvxopt.spmatrix(D_dat, D_row, D_col)
ZtZ_sparse = cvxopt.spmatrix(ZtZ_dat, ZtZ_row, ZtZ_col)

I_sparse = spmatrix(1.0, range(ZtZ.shape[0]), range(ZtZ.shape[0]))
t3 = time.time()
#np.linalg.solve(np.eye(q) + D @ ZtZ, D)
DinvIplusZtZ_sparse = cholmod.splinsolve(D_sparse + ZtZ_sparse, I_sparse)
t2 = time.time()

print(t2-t1)
print(t2-t3)
