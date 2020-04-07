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
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from genTestDat import genTestData2D, prodMats2D
from lib.est2D import *
from lib.npMatrix2d import *
from lib.cvxMatrix2d import *
from lib.PLS import PLS2D, PLS2D_getSigma2, PLS2D_getBeta, PLS2D_getD

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
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData2D(n=8000)

    # Work out number of observations, parameters, random effects, etc
    n = X.shape[0]
    p = X.shape[1]
    q = np.sum(nparams*nlevels)
    qu = np.sum(nparams*(nparams+1)//2)
    r = nlevels.shape[0]

    # Tolerance
    tol = 1e-6

    # Work out factor indices.
    facInds = np.cumsum(nparams*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to dict
    Ddict=dict()
    for k in np.arange(len(nlevels)):

        Ddict[k] = D[facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # -----------------------------------------------------------------------------
    # Display parameters:
    # -----------------------------------------------------------------------------

    print('--------------------------------------------------------------------------')
    print('Test Settings:')
    print('--------------------------------------------------------------------------')
    print('nlevels: ', nlevels)
    print('nparams: ', nparams)
    print('n: ', n, ', p: ', p, ', r: ', r, ', q: ', q, ', tol: ', tol)

    # -----------------------------------------------------------------------------
    # Create empty data frame for results:
    # -----------------------------------------------------------------------------

    # Row indices
    indexVec = np.array(['Time', 'nit'])
    for i in np.arange(p):

        indexVec = np.append(indexVec, 'beta'+str(i+1))

    indexVec = np.append(indexVec, 'sigma2')

    for k in np.arange(r):

        for j in np.arange(nparams[k]*(nparams[k]+1)//2):

            indexVec = np.append(indexVec, 'D'+str(k+1)+','+str(j+1))

    # Construct dataframe
    results = pd.DataFrame(index=indexVec, columns=['Truth', 'PLS', 'FS', 'pFS', 'SFS', 'pSFS', 'cSFS'])

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------

    # Default time and number of iterations
    results.at['Time','Truth']=0
    results.at['nit','Truth']=0

    # Construct parameter vaector
    paramVec_true = beta[:]
    paramVec_true = np.concatenate((paramVec_true,np.array(sigma2).reshape(1,1)),axis=0)

    # Add D to parameter vector
    facInds = np.cumsum(nparams*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])])/sigma2
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # Add results to parameter vector
    for i in np.arange(2,p+qu+3):

        results.at[indexVec[i],'Truth']=paramVec_true[i-2,0]

    #===============================================================================
    # pSFS
    #===============================================================================

    # Get the indices for the individual random factor covariance parameters.
    DkInds = np.zeros(len(nlevels)+1)
    DkInds[0]=np.int(p+1)
    for k in np.arange(len(nlevels)):
        DkInds[k+1] = np.int(DkInds[k] + nparams[k]*(nparams[k]+1)//2)

    # Run Pseudo Simplified Fisher Scoring
    t1 = time.time()
    paramVector_pSFS,_,nit = pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record Time and number of iterations
    results.at['Time','pSFS']=t2-t1
    results.at['nit','pSFS']=nit

    # Record parameters
    for i in np.arange(2,p+qu+3):

        results.at[indexVec[i],'pSFS']=paramVector_pSFS[i-2,0]

    #===============================================================================
    # PLS
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

    # Run Penalized Least Squares
    t1 = time.time()
    estimation = minimize(PLS2D, theta0, args=(ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, I, tinds, rinds, cinds), method='L-BFGS-B', tol=tol)
    
    # Theta parameters 
    theta = estimation['x']

    # Number of iterations
    nit = estimation['nit']

    # Obtain Beta, sigma2 and D
    beta_pls = PLS2D_getBeta(theta, ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, tinds, rinds, cinds)
    sigma2_pls = PLS2D_getSigma2(theta, ZtXtmp, ZtYtmp, XtXtmp, ZtZtmp, XtYtmp, YtXtmp, YtZtmp, XtZtmp, YtYtmp, n, P, I, tinds, rinds, cinds)
    D_pls = PLS2D_getD(theta, tinds, rinds, cinds, sigma2_pls)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','PLS']=t2-t1
    results.at['nit','PLS']=nit

    # Record beta parameters
    for i in range(p):
        results.at[indexVec[i+2],'PLS']=beta_pls[i]

    # Record sigma2 parameter
    results.at['sigma2','PLS']=sigma2_pls[0]
    
    # Indices corresponding to random factors.
    Dinds = np.cumsum(nparams*(nparams+1)//2)+p+3
    Dinds = np.insert(Dinds,0,p+3)

    # Work out vechDk for each random factor
    for k in np.arange(len(nlevels)):
        vechDk = mat2vech2D(np.array(matrix(D_pls[facInds[k]:(facInds[k]+nparams[k]),facInds[k]:(facInds[k]+nparams[k])])))

        # Save parameters
        for j in np.arange(len(vechDk)):
            results.at[indexVec[Dinds[k]+j],'PLS']=vechDk[j,0]

    #===============================================================================
    # cSFS
    #===============================================================================

    # Run Cholesky Simplified Fisher Scoring
    t1 = time.time()
    paramVector_cSFS,_,nit = cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','cSFS']=t2-t1
    results.at['nit','cSFS']=nit
    
    # Save parameters
    for i in np.arange(2,p+qu+3):
        results.at[indexVec[i],'cSFS']=paramVector_cSFS[i-2,0]

    #===============================================================================
    # FS
    #===============================================================================

    # Run Fisher Scoring
    t1 = time.time()
    paramVector_FS,_,nit = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','FS']=t2-t1
    results.at['nit','FS']=nit
    
    # Save parameters
    for i in np.arange(2,p+qu+3):
        results.at[indexVec[i],'FS']=paramVector_FS[i-2,0]

    #===============================================================================
    # SFS
    #===============================================================================

    # Run Simplified Fisher Scoring
    t1 = time.time()
    paramVector_SFS,_,nit = SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','SFS']=t2-t1
    results.at['nit','SFS']=nit

    # Save parameters
    for i in np.arange(2,p+qu+3):
        results.at[indexVec[i],'SFS']=paramVector_SFS[i-2,0]

    #===============================================================================
    # pFS
    #===============================================================================

    # Run Pseudo Fisher Scoring
    t1 = time.time()
    paramVector_pFS,_,nit = pFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None)
    t2 = time.time()

    # Record time and number of iterations
    results.at['Time','pFS']=t2-t1
    results.at['nit','pFS']=nit

    # Save parameters
    for i in np.arange(2,p+qu+3):
        results.at[indexVec[i],'pFS']=paramVector_pFS[i-2,0]

    # Print results
    print(results.to_string())

    # Return results
    return(results)
