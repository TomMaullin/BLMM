import os
import sys
import numpy as np
import pandas as pd
import time
import scipy.sparse
import scipy.sparse.linalg
from scipy import stats
from scipy.optimize import minimize

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
from test.Unit.genTestDat import genTestData2D, prodMats2D
from lib.est2d import *
from lib.npMatrix2d import *

# ==================================================================================
#
# The below simulates random test data and runs all methods described in the LMM 
# paper on the simulated data. It saves all outputs as ... and requires the 
# following inputs:
#
# - OutDir: The output directory.
# - desInd: Integer value between 1 and 3 representing which design to run. The 
#           designs are as follows:
#           - Design 1: nlevels=[25], nraneffs=[2]
#           - Design 2: nlevels=[50,10], nraneffs=[3,2]
#           - Design 3: nlevels=[100,50,10], nraneffs=[4,3,2]
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def sim2D(desInd, OutDir):


    for simInd in range(1,1001):
        
        #===============================================================================
        # Setup
        #===============================================================================

        if desInd==1:
            nlevels = np.array([10])
            nraneffs = np.array([2])
        if desInd==2:
            nlevels = np.array([50,10])
            nraneffs = np.array([3,2])
        if desInd==3:
            nlevels = np.array([100,30,10])
            nraneffs = np.array([4,3,2])

        # Generate test data
        Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D(n=40, p=5, nlevels=nlevels, nraneffs=nraneffs, save=True, simInd=simInd, desInd=desInd, OutDir=OutDir)

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
        # # Display parameters:
        # # -----------------------------------------------------------------------------

        # print('--------------------------------------------------------------------------')
        # print('Test Settings:')
        # print('--------------------------------------------------------------------------')
        # print('nlevels: ', nlevels)
        # print('nraneffs: ', nraneffs)
        # print('n: ', n, ', p: ', p, ', r: ', r, ', q: ', q, ', tol: ', tol)
        # print('--------------------------------------------------------------------------')

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

        # T value p value and Satterthwaite degrees of freedom estimate.
        indexVec = np.append(indexVec,'T')
        indexVec = np.append(indexVec,'p')
        indexVec = np.append(indexVec,'swdf')

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
                
        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_pSFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu]]=T
        results.at[indexVec[p+5+2*qu]]=Pval
        results.at[indexVec[p+6+2*qu]]=df

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

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_cSFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu]]=T
        results.at[indexVec[p+5+2*qu]]=Pval
        results.at[indexVec[p+6+2*qu]]=df

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

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_FS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu]]=T
        results.at[indexVec[p+5+2*qu]]=Pval
        results.at[indexVec[p+6+2*qu]]=df

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

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_SFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu]]=T
        results.at[indexVec[p+5+2*qu]]=Pval
        results.at[indexVec[p+6+2*qu]]=df

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

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_pFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu]]=T
        results.at[indexVec[p+5+2*qu]]=Pval
        results.at[indexVec[p+6+2*qu]]=df

        # Save results
        results.to_csv(os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv'))
    

def timings(desInd, OutDir):

    # Make row indices
    row = ['sim'+str(i) for i in range(1,101)]

    # Make column indices
    col = ['FS','pFS','SFS','pSFS','cSFS','lmer']

    #-----------------------------------------------------------------------------
    # Work out timing stats
    #-----------------------------------------------------------------------------

    # Make timing table
    timesTable = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    timesTable = timesTable.apply(pd.to_numeric)

    for simInd in range(1,1001):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the times
        simTimes = results_table.loc['Time','FS':]

        # Add them to the table
        timesTable.loc['sim'+str(simInd),:]=simTimes

    timesTable.to_csv(os.path.join(OutDir,'timesTable.csv'))


def TstatisticPPplots(desInd, OutDir):

    # Make row indices
    row = ['sim'+str(i) for i in range(1,101)]

    # Make column indices
    col = ['FS','lmer']

    #-----------------------------------------------------------------------------
    # Work out timing stats
    #-----------------------------------------------------------------------------

    # Make timing table
    tTable = pd.DataFrame(index=row, columns=col)
    pTable = pd.DataFrame(index=row, columns=col)
    dfTable = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the tables are numeric
    tTable = tTable.apply(pd.to_numeric)
    pTable = pTable.apply(pd.to_numeric)
    dfTable = dfTable.apply(pd.to_numeric)

    for simInd in range(1,101):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the T, P and df values
        simT = results_table.loc['T',['FS','lmer']]
        simp = results_table.loc['p',['FS','lmer']]
        simdf = results_table.loc['swdf',['FS','lmer']]

        # Add them to the tables
        tTable.loc['sim'+str(simInd),:]=simT
        pTable.loc['sim'+str(simInd),:]=simp
        dfTable.loc['sim'+str(simInd),:]=simdf

    print(tTable.describe())
    print(pTable.describe())
    print(dfTable.describe())


def simT(paramVec, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n):

    # Scalar quantities
    p = XtX.shape[1] # (Number of Fixed Effects parameters)
    q = np.sum(nraneffs*nlevels) # (Total number of random effects)
    qu = np.sum(nraneffs*(nraneffs+1)//2) # (Number of unique random effects)

    # Output beta estimate
    beta = paramVec[0:p,:]  
    
    # Output sigma2 estimate
    sigma2 = paramVec[p:(p+1),:]

    # Get unique D elements (i.e. [vech(D_1),...vech(D_r)])
    vechD = paramVec[(p+1):,:]

    # Get the indices in the paramvector corresponding to D matrices
    IndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # Reconstruct D
    Ddict = dict()

    # D as a dictionary
    for k in np.arange(len(nraneffs)):

        Ddict[k] = vech2mat2D(paramVec[IndsDk[k]:IndsDk[k+1],:])
        
    # Matrix version
    D = np.array([])
    for i in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[i]):
            # Add block
            if i == 0 and j == 0:
                D = Ddict[i]
            else:
                D = scipy.linalg.block_diag(D, Ddict[i])

    # Contrast vector (1 in last place 0 elsewhere)
    L = np.zeros(p)
    L[-1] = 1
    L = L.reshape(1,p)

    # Miscellaneous matrix variables
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Get T statistic
    T = get_T2D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2)

    # Get Satterthwaite estimate of degrees of freedom
    df = get_swdf_T2D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs)

    # Get p value
    # Do this seperately for >0 and <0 to avoid underflow
    if T < 0:
        Pval = 1-stats.t.cdf(T, df)
    else:
        Pval = stats.t.cdf(-T, df)

    return(T,Pval,df)
