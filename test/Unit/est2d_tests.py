import os
import sys
import numpy as np
import cvxopt
from cvxopt import matrix,spmatrix
import pandas as pd
import time
from scipy.optimize import minimize

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from genTestDat import genTestData2D, prodMats2D
from blmm.src.est2d import *
from blmm.src.npMatrix2d import *
from blmm.src.cvxMatrix2d import *
from blmm.src.PeLS import PeLS2D, PeLS2D_getSigma2, PeLS2D_getBeta, PeLS2D_getD

# ==================================================================================
#
# The below function generates a random Linear Mixed Model and then runs parameter 
# estimation using all 2D methods available in the BLMM repository. The resulting
# parameter estimates are combined, along with time taken and number of iterations,
# into a pandas dataframe and the result is printed and returned for comparison.
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def test2D():

    #===============================================================================
    # Setup
    #===============================================================================

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()

    # Work out number of observations, parameters, random effects, etc
    n = X.shape[0]
    p = X.shape[1]
    q = np.sum(nraneffs*nlevels)
    qu = np.sum(nraneffs*(nraneffs+1)//2)
    r = nlevels.shape[0]

    # Tolerance
    tol = 1e-6

    # Work out factor indices.
    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to dict
    Ddict=dict()
    for k in np.arange(len(nlevels)):

        Ddict[k] = D[facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # -----------------------------------------------------------------------------
    # Display parameters:
    # -----------------------------------------------------------------------------

    print('--------------------------------------------------------------------------')
    print('Test Settings:')
    print('--------------------------------------------------------------------------')
    print('nlevels: ', nlevels)
    print('nraneffs: ', nraneffs)
    print('n: ', n, ', p: ', p, ', r: ', r, ', q: ', q, ', tol: ', tol)
    print('--------------------------------------------------------------------------')

    # -----------------------------------------------------------------------------
    # Create empty data frame for results:
    # -----------------------------------------------------------------------------

    # Row indices
    indexVec = np.array(['Time', 'nit', 'llh'])
    for i in np.arange(p):

        indexVec = np.append(indexVec, 'beta'+str(i+1))

    # Sigma2
    indexVec = np.append(indexVec, 'sigma2')

    # Dk
    for k in np.arange(r):
        for j in np.arange(nraneffs[k]*(nraneffs[k]+1)//2):
            indexVec = np.append(indexVec, 'D'+str(k+1)+','+str(j+1))

    # Sigma2*Dk
    for k in np.arange(r):
        for j in np.arange(nraneffs[k]*(nraneffs[k]+1)//2):
            indexVec = np.append(indexVec, 'sigma2*D'+str(k+1)+','+str(j+1))

    # Construct dataframe
    results = pd.DataFrame(index=indexVec, columns=['Truth', 'PeLS', 'FS', 'pFS', 'SFS', 'pSFS', 'cSFS'])

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------

    # Default time and number of iterations
    results.at['Time','Truth']=0
    results.at['nit','Truth']=0

    # Construct parameter vector
    paramVec_true = beta[:]
    paramVec_true = np.concatenate((paramVec_true,np.array(sigma2).reshape(1,1)),axis=0)

    # Add D to parameter vector
    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # Add results to parameter vector
    for i in np.arange(3,p+qu+4):

        results.at[indexVec[i],'Truth']=paramVec_true[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'Truth']=paramVec_true[p,0]*paramVec_true[i-3,0]

    # Matrices needed for
    Zte = ZtY - ZtX @ beta
    ete = ssr2D(YtX, YtY, XtX, beta)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # True log likelihood
    llh = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]-n/2*np.log(np.pi)
    results.at['llh','Truth']=llh

    print('Truth Saved')

    #===============================================================================
    # pSFS
    #===============================================================================

    # Get the indices for the individual random factor covariance parameters.
    DkInds = np.zeros(len(nlevels)+1)
    DkInds[0]=np.int(p+1)
    for k in np.arange(len(nlevels)):
        DkInds[k+1] = np.int(DkInds[k] + nraneffs[k]*(nraneffs[k]+1)//2)

    # Run Pseudo Simplified Fisher Scoring
    t1 = time.time()
    paramVector_pSFS,_,nit,llh = pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record Time and number of iterations
    results.at['Time','pSFS']=t2-t1
    results.at['nit','pSFS']=nit
    results.at['llh','pSFS']=llh-n/2*np.log(np.pi)

    # Record parameters
    for i in np.arange(3,p+qu+4):

        results.at[indexVec[i],'pSFS']=paramVector_pSFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'pSFS']=paramVector_pSFS[p,0]*paramVector_pSFS[i-3,0]

    print('pSFS Saved')
    
    #===============================================================================
    # PeLS
    #===============================================================================

    # Convert matrices to cvxopt format.
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
    for i in np.arange(len(nraneffs)):
        theta0 = np.hstack((theta0, mat2vech2D(np.eye(nraneffs[i])).reshape(np.int64(nraneffs[i]*(nraneffs[i]+1)/2))))
      
    # Obtain a random Lambda matrix with the correct sparsity for the permutation vector
    tinds,rinds,cinds=get_mapping2D(nlevels, nraneffs)
    Lam=mapping2D(np.random.randn(theta0.shape[0]),tinds,rinds,cinds)

    # Obtain Lambda'Z'ZLambda
    LamtZtZLam = spmatrix.trans(Lam)*cvxopt.sparse(matrix(ZtZtmp))*Lam

    # Identity (Actually quicker to calculate outside of estimation)
    I = spmatrix(1.0, range(Lam.size[0]), range(Lam.size[0]))

    # Obtaining permutation for PeLS
    P=cvxopt.amd.order(LamtZtZLam)

    # Run Penalized Least Squares
    t1 = time.time()
    estimation = minimize(PeLS2D, theta0, args=(ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, I, tinds, rinds, cinds), method='L-BFGS-B', tol=tol)
    
    # llh
    llh = -estimation['fun']

    # Theta parameters 
    theta = estimation['x']

    # Number of iterations
    nit = estimation['nit']

    # Obtain Beta, sigma2 and D
    beta_pls = PeLS2D_getBeta(theta, ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, tinds, rinds, cinds)
    sigma2_pls = PeLS2D_getSigma2(theta, ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, I, tinds, rinds, cinds)
    D_pls = PeLS2D_getD(theta, tinds, rinds, cinds, sigma2_pls)
    t2 = time.time()

    # Record time, number of iterations and log likelihood
    results.at['Time','PeLS']=t2-t1
    results.at['nit','PeLS']=nit
    results.at['llh','PeLS']=llh

    # Record beta parameters
    for i in range(p):
        results.at[indexVec[i+3],'PeLS']=beta_pls[i]

    # Record sigma2 parameter
    results.at['sigma2','PeLS']=sigma2_pls[0]
    
    # Indices corresponding to random factors.
    Dinds = np.cumsum(nraneffs*(nraneffs+1)//2)+p+4
    Dinds = np.insert(Dinds,0,p+4)

    # Work out vechDk for each random factor
    for k in np.arange(len(nlevels)):
        vechDk = mat2vech2D(np.array(matrix(D_pls[facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])))

        # Save parameters
        for j in np.arange(len(vechDk)):
            results.at[indexVec[Dinds[k]+j],'PeLS']=vechDk[j,0]/sigma2_pls[0]
            results.at[indexVec[Dinds[k]+qu+j],'PeLS']=vechDk[j,0]

    print('PeLS Saved')
    
    #===============================================================================
    # cSFS
    #===============================================================================

    # Run Cholesky Simplified Fisher Scoring
    t1 = time.time()
    paramVector_cSFS,_,nit,llh = cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','cSFS']=t2-t1
    results.at['nit','cSFS']=nit
    results.at['llh','cSFS']=llh-n/2*np.log(np.pi)
    
    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'cSFS']=paramVector_cSFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'cSFS']=paramVector_cSFS[p,0]*paramVector_cSFS[i-3,0]
        
    print('cSFS Saved')
    
    #===============================================================================
    # FS
    #===============================================================================

    # Run Fisher Scoring
    t1 = time.time()
    paramVector_FS,_,nit,llh = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','FS']=t2-t1
    results.at['nit','FS']=nit
    results.at['llh','FS']=llh-n/2*np.log(np.pi)
    
    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'FS']=paramVector_FS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'FS']=paramVector_FS[p,0]*paramVector_FS[i-3,0]

    print('FS Saved')

    #===============================================================================
    # SFS
    #===============================================================================

    # Run Simplified Fisher Scoring
    t1 = time.time()
    paramVector_SFS,_,nit,llh = SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','SFS']=t2-t1
    results.at['nit','SFS']=nit
    results.at['llh','SFS']=llh-n/2*np.log(np.pi)

    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'SFS']=paramVector_SFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'SFS']=paramVector_SFS[p,0]*paramVector_SFS[i-3,0]

    print('SFS Saved')
    
    #===============================================================================
    # pFS
    #===============================================================================

    # Run Pseudo Fisher Scoring
    t1 = time.time()
    paramVector_pFS,_,nit,llh = pFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','pFS']=t2-t1
    results.at['nit','pFS']=nit
    results.at['llh','pFS']=llh-n/2*np.log(np.pi)

    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'pFS']=paramVector_pFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'pFS']=paramVector_pFS[p,0]*paramVector_pFS[i-3,0]

    print('pFS Saved')
    
    # Print results
    print(results.to_string())
    
    # Return results
    return(results)
