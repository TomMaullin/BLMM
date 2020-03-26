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

# Get the product matrices
XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

print(nlevels)
print(nparams)

t1 = time.time()
paramVector_cSFS,_ = cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print(t2-t1)

t1 = time.time()
paramVector_SFS,_ = SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print(t2-t1)

t1 = time.time()
paramVector_pSFS,_ = pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print(t2-t1)


t1 = time.time()
paramVector_pFS,_ = pFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print(t2-t1)

t1 = time.time()
paramVector_FS,_ = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
t2 = time.time()
print(t2-t1)

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
print(t2-t1)
invDupMatdict = dict()
for i in np.arange(len(nparams)):
  invDupMatdict[i] = invDupMat2D(nparams[i])

# Work out D indices (there is one block of D per level)
Dinds = np.zeros(np.sum(nlevels)+1)
counter = 0
for k in np.arange(len(nparams)):
  for j in np.arange(nlevels[k]):
    Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nparams)))[k] + nparams[k]*j
    counter = counter + 1
      
# Last index will be missing so add it
Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nparams[-1]

# Make sure indices are ints
Dinds = np.int64(Dinds)

# Work out the total number of paramateres
tnp = np.int32(p + 1 + np.sum(nparams*(nparams+1)//2))

FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + p + 1)
FishIndsDk = np.insert(FishIndsDk,0,p+1)

def negllh_hybrid(paramVec):

  beta = paramVec[0:p].reshape(p,1)
  sigma2 = np.absolute(paramVec[p])
#  print(sigma2,'sig')


  Zte = ZtY - ZtX @ beta
  ete = YtY - 2*YtX @ beta + beta.transpose() @ XtX @ beta

  D = scipy.sparse.lil_matrix((q,q))
  counter = 0
  for k in np.arange(len(nparams)):
    for j in np.arange(nlevels[k]):
      
      D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = makeDnnd2D(vech2mat2D(paramVec[FishIndsDk[k]:FishIndsDk[k+1]]))
      counter = counter + 1
        
  IplusZtZD = np.eye(q) + (ZtZ @ D)
  DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))


  return(llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D))

def dl(paramVec):

  beta = paramVec[0:p].reshape(p,1)
  sigma2 = np.absolute(paramVec[p])


  Zte = ZtY - ZtX @ beta
  Xte = XtY - XtX @ beta
  ete = YtY - 2*YtX @ beta + beta.transpose() @ XtX @ beta

  D = scipy.sparse.lil_matrix((q,q))
  counter = 0
  for k in np.arange(len(nparams)):
    for j in np.arange(nlevels[k]):

      D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = makeDnnd2D(vech2mat2D(paramVec[FishIndsDk[k]:FishIndsDk[k+1]]))
      counter = counter + 1
        
  IplusZtZD = np.eye(q) + (ZtZ @ D)
  DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))

  
  # Derivative wrt beta
  dldB = get_dldB2D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte)

  # Derivative wrt sigma^2
  dldsigma2 = get_dldsigma22D(n, ete, Zte, sigma2, DinvIplusZtZD)
    
  # For each factor, factor k, work out dl/dD_k
  dldDdict = dict()
  for k in np.arange(len(nparams)):

    # Store it in the dictionary
    dldDdict[k],_ = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)

  derivVector = np.concatenate((dldB, dldsigma2))
  for k in np.arange(len(nparams)):
    derivVector = np.concatenate((derivVector, mat2vech2D(dldDdict[k])))

  derivVal = derivVector.reshape(p+1+np.sum(nparams*(nparams+1)//2))

  return(derivVal)

def H(paramVec):

  beta = paramVec[0:p].reshape(p,1)
  sigma2 = np.absolute(paramVec[p])

  Zte = ZtY - ZtX @ beta
  ete = YtY - 2*YtX @ beta + beta.transpose() @ XtX @ beta

  D = scipy.sparse.lil_matrix((q,q))
  counter = 0
  for k in np.arange(len(nparams)):
    for j in np.arange(nlevels[k]):

      D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = makeDnnd2D(vech2mat2D(paramVec[FishIndsDk[k]:FishIndsDk[k+1]]))
      counter = counter + 1
        
  IplusZtZD = np.eye(q) + (ZtZ @ D)
  DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))


  # Construct the Fisher Information matrix
  # ----------------------------------------------------------------------------
  FisherInfoMat = np.zeros((tnp,tnp))

  # Add dl/dbeta covariance
  FisherInfoMat[np.ix_(np.arange(p),np.arange(p))] = get_covdldbeta2D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2)

  # Add dl/dsigma2 covariance
  covdldsigma2 = n/(2*(sigma2**2))
  FisherInfoMat[p,p] = covdldsigma2

  # Add dl/dsigma2 dl/dD covariance
  for k in np.arange(len(nparams)):

    # Assign to the relevant block
    covdldDksigma2,_ = get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, ZtZmat=None)

    # Assign to the relevant block
    FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldDksigma2.reshape(FishIndsDk[k+1]-FishIndsDk[k])
    FisherInfoMat[FishIndsDk[k]:FishIndsDk[k+1],p] = FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]].transpose()

  # Add dl/dD covariance
  for k1 in np.arange(len(nparams)):

    for k2 in np.arange(k1+1):

      IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
      IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

      # Get covariance between D_k1 and D_k2 
      FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],_ = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict,perm=None)      
      FisherInfoMat[np.ix_(IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(IndsDk1, IndsDk2)].transpose()

  FisherInfoMat = forceSym2D(FisherInfoMat)

  return(FisherInfoMat)

#Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

pvec = np.hstack((np.ones(p+1),theta0))
print(negllh_hybrid(pvec).shape)
print(pvec.shape)
print(dl(pvec).shape)
print(H(pvec).shape)

# SANDBOX OPTIMIZERS
t1 = time.time()
paramVec_hybrid = minimize(negllh_hybrid, np.hstack((np.ones(p+1),theta0)), jac=dl, hess=H, method='trust-krylov', tol=1e-7)['x']
t2 = time.time()
print(t2-t1)
