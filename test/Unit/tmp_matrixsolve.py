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

Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData2D()#,nparams=np.array([2]),nlevels=np.array([30]))
n = X.shape[0]
p = X.shape[1]

IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()

k = 0

Ai = dict()
for i in np.arange(nlevels[k]):
    Iki = faclev_indices2D(k, i, nlevels, nparams)
    Zki = Z[:,Iki]
    Ai[i] = Zki.transpose() @ scipy.linalg.sqrtm(np.linalg.inv(IplusZDZt))

Aij = dict()
for i in np.arange(nlevels[k]):
    for j in np.arange(nlevels[k]):
        Aij[str(i)+str(j)] = Ai[i] @ Ai[j].transpose()

        if i == 0 and j == 0:

            kronSum = np.kron(Aij[str(i)+str(j)],Aij[str(i)+str(j)])
        
        else:

            kronSum = kronSum + np.kron(Aij[str(i)+str(j)],Aij[str(i)+str(j)])

        if i == j:

            if i == 0:

                otherSum = Aij[str(i)+str(j)]
                
            else:

                otherSum = otherSum + Aij[str(i)+str(j)]

print(np.linalg.inv(kronSum) @ mat2vec2D(otherSum))

ZtZ = Z.transpose() @ Z
iZtZ = np.linalg.inv(ZtZ)

for i in np.arange(nlevels[k]):

    Iki = faclev_indices2D(k, i, nlevels, nparams)
    Zki = Z[:,Iki]

    kTerm = (Zki.transpose() @ Z @ iZtZ).transpose()
    
    if i == 0:

        runningSum = np.kron(kTerm,kTerm)

    else:

        runningSum = runningSum + np.kron(kTerm,kTerm)

b = np.linalg.pinv(runningSum) @ mat2vec2D(iZtZ + D)
print(b)
    


