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
from lib.PLS import PLS2D, PLS2D_getSigma2, PLS2D_getBeta, PLS2D_getD
from scipy.optimize import minimize

Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()#nparams=np.array([2,2]),nlevels=np.array([2000,30]))
n = X.shape[0]
p = X.shape[1]
q = np.sum(nparams*nlevels)
tol = 1e-6

# Get the product matrices
XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

print(nparams)
print(dupMat2D(nparams[0]).shape)
# New empty D dict
Ddict = dict()

# Work out the indices in D where a new block Dk appears
Dinds = np.cumsum(nlevels*nparams)
Dinds = np.insert(Dinds,0,0)


# Work out Dk for each factor, factor k 
for k in np.arange(nlevels.shape[0]):

    # Add Dk to the dict
    Ddict[k] = D[Dinds[k]:(Dinds[k]+nparams[k]),Dinds[k]:(Dinds[k]+nparams[k])]



# Decide on a random factor
k = np.random.randint(0,nparams.shape[0])

print(np.linalg.cholesky(Ddict[k]))


lam = np.linalg.cholesky(Ddict[k])


# Want

# L

def elimMat2D(n):

    # Work out indices of lower triangular matrix
    tri_row, tri_col = np.tril_indices(n)

    # Translate these into the column indices we need
    elim_col = np.sort(tri_col*n+tri_row)

    # The row indices are just 1 to n(n+1)/2
    elim_row = np.arange(n*(n+1)//2)

    # We need to put ones in
    elim_dat = np.ones(n*(n+1)//2)

    # Construct the elimination matrix
    elim=scipy.sparse.csr_matrix((elim_dat,(elim_row,elim_col)))

    # Return 
    return(elim)

def vechTri2mat2D(vech):

    # Return lower triangular
    return(np.tril(vech2mat2D(vech)))

def mat2vechTri2D(mat):

    # Return vech
    return(mat2vech2D(mat))

x = np.random.randn(8,8)
x = x @ x.transpose()

vechx = mat2vech2D(x)

ltrixmat = vechTri2mat2D(vechx)

print(vechx)
test = mat2vech2D(ltrixmat)
print(test)
print('hfiwan')
t1 = time.time()
elimMat2D(8)
t2 = time.time()
print(t2-t1)

m = 20

# Make random lower triangular matrix
a = np.tril(np.random.randn(m**2).reshape(m,m))
print(a)

# Convert to vec
avec = mat2vec2D(a)
print(avec)

# Multiply by elimination matrix
print(elimMat2D(m) @ avec)

print(avec[avec!=0].reshape(avec[avec!=0].shape[0],1)-elimMat2D(m) @ avec)

# Need elim mat
L = elimMat2D(nparams[k])

# Times I_qk^2 + K_qk
#L @ (scipy.sparse.identity(nparams[k]**2) + comMat2D(nparams[k],nparams[k]))
print(comMat2D(nparams[k],nparams[k]).toarray())
print(nparams[k])

# Times (lam kron I_qk)
#L @ (scipy.sparse.identity(nparams[k]**2) + comMat2D(nparams[k],nparams[k])) @ scipy.sparse.kron(lam,np.eye(nparams[k]))

# Times dupMat_qk
L @ (scipy.sparse.identity(nparams[k]**2) + comMat2D(nparams[k],nparams[k])) @ scipy.sparse.kron(lam,np.eye(nparams[k])) @ dupMat2D(nparams[k])

