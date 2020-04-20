import os
import sys
import numpy as np
import pandas as pd
import time
import scipy.sparse
import scipy.sparse.linalg
import sys

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.npMatrix2d import *
from genTestDat import prodMats2D, genTestData2D

Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData2D(n=100,nraneffs=np.array([2,1]),nlevels=np.array([4,4]))
n = X.shape[0]
p = X.shape[1]
q = Z.shape[1]

print(np.mean(D))
print(sigma2)

# Get the product matrices
XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

ZtZs = scipy.sparse.csr_matrix(ZtZ)

k = 0
    

# Work out Zte
Zte = ZtY - ZtX @ beta




# Work it out current way:


# Duplication matrices
# ------------------------------------------------------------------------------
invDupMatdict = dict()
for i in np.arange(len(nparams)):
    invDupMatdict[i] = invDupMat2D(nparams[i])

DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

# This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
ZtZmatdict = dict()
for i in np.arange(len(nparams)):
    ZtZmatdict[i] = None

# This will hold the permutations needed for the covariance between the
# derivatives with respect to k
permdict = dict()
for i in np.arange(len(nparams)):
    permdict[str(i)] = None
    
# Work out derivative
if ZtZmatdict[k] is None:
    dldD,ZtZmatdict[k] = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
else:
    dldD,_ = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

      
# Work out update amount
if permdict[str(k)] is None:
    covdldDk,permdict[str(k)] = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True, perm=None)
else:
    covdldDk,_ = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True, perm=permdict[str(k)])

print(nparams[k])
print(covdldDk.shape)
print(mat2vec2D(dldD).shape)
# Work out update amount
update = forceSym2D(np.linalg.inv(covdldDk)) @ mat2vec2D(dldD)

print(update)


# ===============================================================================

# Work out sum R kron R

IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
invIplusZDZt = np.linalg.inv(IplusZDZt)
e = Y - X @ beta

for i in np.arange(nlevels[k]):

    for j in np.arange(nlevels[k]):

        # Work out Z_j
        Ikj = faclev_indices2D(k, j, nlevels, nparams)
        Zkj = Z[:,Ikj]

        # Work out Z_i
        Iki = faclev_indices2D(k, i, nlevels, nparams)
        Zki = Z[:,Iki]
        
        Rkij = Zkj.transpose() @ invIplusZDZt @ Zki

        if i==j:

            Kki = Zki.transpose() @ invIplusZDZt @ e @ e.transpose() @ invIplusZDZt @ Zki
            
            if i ==0:

                Rksum = Rkij
                Kksum = Kki

            else:

                Rksum = Rkij + Rksum
                Kksum = Kki + Kksum

                


print(sigma2)
print('marker1')
        
#==============================================================================
#==============================================================================


uMat = vec2mat2D(update)

for i in np.arange(nlevels[k]):

    for j in np.arange(nlevels[k]):

        # Work out Z_j
        Ikj = faclev_indices2D(k, j, nlevels, nparams)
        Zkj = Z[:,Ikj]

        # Work out Z_i
        Iki = faclev_indices2D(k, i, nlevels, nparams)
        Zki = Z[:,Iki]
        
        Rkij = Zkj.transpose() @ invIplusZDZt @ Zki

        if i==0 and j==0:

            RHS = Rkij.transpose() @ uMat @ Rkij

        else:

            RHS = RHS + Rkij.transpose() @ uMat @ Rkij

LHS = Kksum/sigma2 - Rksum

print(np.allclose(RHS,LHS))

#==============================================================================


