import os
import sys
import numpy as np
import cvxopt
from cvxopt import matrix,spmatrix
import pandas as pd
import os
import time
import scipy.sparse
import scipy.sparse.linalg
import sys

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.tools2d import *
from unitTests2D import genTestData, prodMats
from lib.FS2D import FS2D
from lib.SFS2D import SFS2D
from lib.pFS2D import pFS2D
from lib.pSFS2D import pSFS2D
from lib.cSFS2D import cSFS2D
from lib.PLS import PLS2D, PLS2D_getSigma2, PLS2D_getBeta, PLS2D_getD
from scipy.optimize import minimize

Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()#nparams=np.array([2,2]),nlevels=np.array([2000,30]))
n = X.shape[0]
p = X.shape[1]
q = np.sum(nparams*nlevels)
tol = 1e-6

facInds = np.cumsum(nparams*nlevels)
facInds = np.insert(facInds,0,0)

# Convert D to dict
Ddict=dict()
for k in np.arange(len(nlevels)):

  Ddict[k] = D[facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])]

# Get the product matrices
XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

print(nlevels)
print(nparams)

# Get the paramvec indices
bInds = np.arange(p)
sigInd = p
DkInds = np.zeros(len(nlevels)+1)
DkInds[0]=np.int(p+1)
for k in np.arange(len(nlevels)):
  DkInds[k+1] = np.int(DkInds[k] + nparams[k]*(nparams[k]+1)//2)

t1 = time.time()
paramVector_pSFS,_ = pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print('Time (pSFS): ', t2-t1)
print('Method: pSFS')
print('beta:')
print('   True: ', beta)
print('   Est: ', paramVector_pSFS[bInds])
print('Sigma^2:')
print('   True: ', sigma2)
print('   Est: ', paramVector_pSFS[sigInd])
for k in np.arange(len(nlevels)):
  print('D (', k, '): ')
  print('   True: ', Ddict[k])
  print('   Est: ', paramVector_pSFS[sigInd]*vech2mat2D(paramVector_pSFS[np.int(DkInds[k]):np.int(DkInds[k+1])]))



#===============================================================================
# PLS

XtZtmp = matrix(XtZ)
ZtXtmp = matrix(ZtX)
ZtZtmp = cvxopt.sparse(matrix(ZtZ))
XtXtmp = matrix(XtX)

XtYtmp = matrix(XtY) 
ZtYtmp = matrix(ZtY) 
YtYtmp = matrix(YtY) 
YtZtmp = matrix(YtZ)
YtXtmp = matrix(YtX)
  
# Initial theta value. Bates (2005) suggests using [vech(I_q1),...,vech(I_qr)] where I is the identity matrix
theta0 = np.array([])
for i in np.arange(len(nparams)):
  theta0 = np.hstack((theta0, mat2vech2D(np.eye(nparams[i])).reshape(np.int64(nparams[i]*(nparams[i]+1)/2))))
  
# Obtain a random Lambda matrix with the correct sparsity for the permutation vector
tinds,rinds,cinds=get_mapping2D(nlevels, nparams)
Lam=mapping2D(np.random.randn(theta0.shape[0]),tinds,rinds,cinds)

# Obtain Lambda'Z'ZLambda
LamtZtZLam = spmatrix.trans(Lam)*cvxopt.sparse(matrix(ZtZtmp))*Lam


# Identity (Actually quicker to calculate outside of estimation)
I = spmatrix(1.0, range(Lam.size[0]), range(Lam.size[0]))

# Obtaining permutation for PLS
P=cvxopt.amd.order(LamtZtZLam)

t1 = time.time()
theta = minimize(PLS2D, theta0, args=(ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, I, tinds, rinds, cinds), method='L-BFGS-B', tol=1e-7)['x']
beta_pls = PLS2D_getBeta(theta, ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, tinds, rinds, cinds)
sigma2_pls = PLS2D_getSigma2(theta, ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, I, tinds, rinds, cinds)
D_pls = PLS2D_getD(theta, tinds, rinds, cinds, sigma2_pls)
t2 = time.time()
print('Time (PLS): ', t2-t1)
print('Method: PLS')
print('beta:')
print('   True: ', beta)
print('   Est: ', beta_pls)
print('Sigma^2:')
print('   True: ', sigma2)
print('   Est: ', sigma2_pls)
for k in np.arange(len(nlevels)):
  print('D (', k, '): ')
  print('   True: ', Ddict[k])
  print('   Est: ', D_pls[facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])])


t1 = time.time()
paramVector_cSFS,_ = cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print('Time (cSFS): ', t2-t1)
print('Method: cSFS')
print('beta:')
print('   True: ', beta)
print('   Est: ', paramVector_cSFS[bInds])
print('Sigma^2:')
print('   True: ', sigma2)
print('   Est: ', paramVector_cSFS[sigInd])
for k in np.arange(len(nlevels)):
  print('D (', k, '): ')
  print('   True: ', Ddict[k])
  print('   Est: ', paramVector_cSFS[sigInd]*vech2mat2D(paramVector_cSFS[np.int(DkInds[k]):np.int(DkInds[k+1])]))



t1 = time.time()
paramVector_FS,_ = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print('Time (FS): ', t2-t1)
print('Method: FS')
print('beta:')
print('   True: ', beta)
print('   Est: ', paramVector_FS[bInds])
print('Sigma^2:')
print('   True: ', sigma2)
print('   Est: ', paramVector_FS[sigInd])
for k in np.arange(len(nlevels)):
  print('D (', k, '): ')
  print('   True: ', Ddict[k])
  print('   Est: ', paramVector_FS[sigInd]*vech2mat2D(paramVector_FS[np.int(DkInds[k]):np.int(DkInds[k+1])]))



t1 = time.time()
paramVector_SFS,_ = SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print('Time (SFS): ', t2-t1)
print('Method: SFS')
print('beta:')
print('   True: ', beta)
print('   Est: ', paramVector_SFS[bInds])
print('Sigma^2:')
print('   True: ', sigma2)
print('   Est: ', paramVector_SFS[sigInd])
for k in np.arange(len(nlevels)):
  print('D (', k, '): ')
  print('   True: ', Ddict[k])
  print('   Est: ', paramVector_SFS[sigInd]*vech2mat2D(paramVector_SFS[np.int(DkInds[k]):np.int(DkInds[k+1])]))

DkInds2 = np.zeros(len(nlevels)+1)
DkInds2[0]=np.int(p+1)
for k in np.arange(len(nlevels)):
  DkInds2[k+1] = np.int(DkInds2[k] + nparams[k]**2)

t1 = time.time()
paramVector_pFS,_ = pFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print('Time (pFS): ', t2-t1)
print('Method: pFS')
print('beta:')
print('   True: ', beta)
print('   Est: ', paramVector_pFS[bInds])
print('Sigma^2:')
print('   True: ', sigma2)
print('   Est: ', paramVector_pFS[sigInd])
for k in np.arange(len(nlevels)):
  print('D (', k, '): ')
  print('   True: ', Ddict[k])
  print('   Est: ', paramVector_pFS[sigInd]*vec2mat2D(paramVector_pFS[np.int(DkInds2[k]):np.int(DkInds2[k+1])]))
