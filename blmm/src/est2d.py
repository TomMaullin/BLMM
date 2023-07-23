import time
import numpy as np
import scipy
from blmm.src.npMatrix3d import *
from blmm.src.npMatrix2d import *

# ============================================================================
#
# This file contains code for all Fisher Scoring based (univariate/"one-
# voxel") parameter estimation methods developed during the course of the BLMM
# project. The methods given here are:
#
# - `FS`: Fisher Scoring
# - `pFS`: Pseudo-Fisher Scoring
# - `SFS`: Simplified Fisher Scoring
# - `pFS`: Pseudo-Simplified Fisher Scoring
# - `cSFS`: Cholesky Simplified Fisher 
# 
# The PeLS algorithm currently is not included here, but instead can be found 
# in the file "PeLS.py".
#
# ----------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited 07/04/2020)
#
# ============================================================================


# ============================================================================
# 
# This below function performs Cholesky Fisher Scoring for the Linear Mixed
# Model. It is based on the update rules:
#
#                       beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)
#
#                             sigma2 = e'V^(-1)e/n
#
#                              for k in {1,...,r};
#  vechTri(Chol_k) = \theta_f + lam*I(vechTri(Chol_k))^(-1) (dl/dvechTri(Chol_k))
#
# Where:
#  - chol_k is the lower triangular cholesky factor of D_k
#  - vechTri(A) is the vector of lower triangular elements of A.
#  - lam is a scalar stepsize.
#  - I(vechTri(Chol_k)) is the Fisher Information matrix of vechTri(Chol_k).
#  - dl/dvechTri(Chol_k) is the derivative of the log likelihood of 
#    (beta, sigma^2, vechTri(Chol_1),...vechTri(Chol_r)) with respect
#    to vechTri(Chol_k). 
#  - e is the residual vector (e=Y-X\beta)
#  - V is the matrix (I+ZDZ').
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X.
#  - `XtY`: X transpose multiplied by Y.
#  - `XtZ`: X transpose multiplied by Z. 
#  - `YtX`: Y transpose multiplied by X.
#  - `YtY`: Y transpose multiplied by Y.
#  - `YtZ`: Y transpose multiplied by Z.
#  - `ZtX`: Z transpose multiplied by X.
#  - `ZtY`: Z transpose multiplied by Y.
#  - `ZtZ`: Z transpose multiplied by Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations.
#  - `init_paramVector`: (Optional) initial estimates of the parameter vector.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `paramVector`: The parameter vector (beta,sigma2,vech(D_1),...,vech(D_r))
#  - `bvals`: Estimates of the random effects vector, b.
#  - `nit`: The number of iterations taken to converge.
#
# ============================================================================
def cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None):
    
    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[0]

    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of parameters
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # ------------------------------------------------------------------------------
    # Work out D indices (there is one block of D per level)
    # ------------------------------------------------------------------------------
    Dinds = np.zeros(np.sum(nlevels)+1)
    counter = 0

    # Loop through and add each index
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1
            
    # Last index will be missing so add it
    Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]
    
    # Make sure indices are ints
    Dinds = np.int64(Dinds)

    # ------------------------------------------------------------------------------
    # Duplication, Commutation and Elimination matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    elimMatdict = dict()
    comMatdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()
        comMatdict[i] = comMat2D(nraneffs[i],nraneffs[i])
        elimMatdict[i] = elimMat2D(nraneffs[i])
        
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------
    # If they have been specified as inputs use those.
    if init_paramVector is not None:

        # beta and sigma2 initial values
        beta = init_paramVector[0:p]
        sigma2 = init_paramVector[p:(p+1)][0,0]
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Initial cholesky decomposition and D.
        Ddict = dict()
        cholDict = dict()
        for k in np.arange(len(nraneffs)):

            Ddict[k] = makeDpd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
            cholDict[k] = np.linalg.cholesky(Ddict[k])
        
        # Matrix version
        D = scipy.sparse.lil_matrix((q,q))
        counter = 0
        for k in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):

                D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1
        D = D.toarray()

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Z'e, needed for first iteration
        Zte = ZtY - (ZtX @ beta)

    # Otherwise use the closed form initial estimates
    else:

        # Inital beta
        beta = initBeta2D(XtX, XtY)

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Z'e
        Zte = ZtY - (ZtX @ beta)
            
        # Inital cholesky decomposition and D
        Ddict = dict()
        cholDict = dict()
        for k in np.arange(len(nraneffs)):

            # We just initialize to identity for cholesky.
            cholDict[k] = np.eye(nraneffs[k])
            Ddict[k] = np.eye(nraneffs[k])

        # Matrix version
        D = scipy.sparse.lil_matrix((q,q))
        t1 = time.time()
        counter = 0
        for k in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[k]):

                D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1
        D = D.toarray()

    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1)
    # ------------------------------------------------------------------------------
    DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = 1
    
    # Initial log likelihoods
    llhprev = np.inf
    llhcurr = -np.inf
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k in np.arange(len(nraneffs)):
        permdict[str(k)] = None

    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    # Number of iterations
    nit = 0
    while np.abs(llhprev-llhcurr)>tol:

        # Change current likelihood to previous
        llhprev = llhcurr

        # Update number of iterations
        nit = nit+1

        #---------------------------------------------------------------------------
        # Update beta
        #---------------------------------------------------------------------------
        beta = np.linalg.solve(XtX - XtZ @ DinvIplusZtZD @ ZtX, XtY - XtZ @ DinvIplusZtZD @ ZtY)
        
        #---------------------------------------------------------------------------
        # Update sigma2
        #---------------------------------------------------------------------------
        sigma2 = 1/n*(ete - Zte.transpose() @ DinvIplusZtZD @ Zte)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary
        
        #---------------------------------------------------------------------------
        # Update Cholesky factor
        #---------------------------------------------------------------------------
        counter = 0
        # Loop though unique blocks of D updating one at a time
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Calculate derivative with respect to D_k
            #-----------------------------------------------------------------------
            # Work out derivative
            if ZtZmatdict[k] is None:
                dldD,ZtZmatdict[k] = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
            else:
                dldD,_ = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

            #-----------------------------------------------------------------------
            # Calculate covariance of derivative with respect to D_k
            #-----------------------------------------------------------------------
            if permdict[str(k)] is None:
                covdldDk,permdict[str(k)] = get_covdldDk1Dk22D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=None)
            else:
                covdldDk,_ = get_covdldDk1Dk22D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=permdict[str(k)])


            #-----------------------------------------------------------------------
            # Transform to derivative with respect to chol_k
            #-----------------------------------------------------------------------
            # We need to modify by multiplying by this matrix to obtain the cholesky derivative.
            chol_mod = elimMatdict[k] @ (scipy.sparse.identity(nraneffs[k]**2) + comMatdict[k]) @ scipy.sparse.kron(cholDict[k],np.eye(nraneffs[k])) @ dupMatTdict[k].transpose()
            
            # Transform to cholesky
            dldcholk = chol_mod.transpose() @ dupMatTdict[k] @ mat2vec2D(dldD)

            #-----------------------------------------------------------------------
            # Transform to covariance of derivative with respect to chol_k
            #-----------------------------------------------------------------------
            covdldcholk = chol_mod.transpose() @ covdldDk @ chol_mod

            #-----------------------------------------------------------------------
            # Perform update
            #-----------------------------------------------------------------------
            update = lam*np.linalg.solve(forceSym2D(covdldcholk), dldcholk)
        
            #-----------------------------------------------------------------------
            # Update D_k and chol_k
            #-----------------------------------------------------------------------
            cholDict[k] = vechTri2mat2D(mat2vechTri2D(cholDict[k]) + update)
            Ddict[k] = cholDict[k] @ cholDict[k].transpose()

            # Add D_k back into D and recompute DinvIplusZtZD
            for j in np.arange(nlevels[k]):

                D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1
            
            #-----------------------------------------------------------------------
            # Obtain D(I+Z'ZD)^(-1)
            #-----------------------------------------------------------------------
            DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

        # --------------------------------------------------------------------------
        # Matrices for next iteration
        # --------------------------------------------------------------------------
        # Recalculate Zte and ete
        Zte = ZtY - (ZtX @ beta)

        # Sum of squared residuals
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Check sigma2 hasn't hit a boundary
        if sigma2<0:
            sigma2=1e-10

        #---------------------------------------------------------------------------
        # Update the step size and log likelihood
        #---------------------------------------------------------------------------
        llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
        if llhprev>llhcurr:
            lam = lam/2

    #-------------------------------------------------------------------------------
    # Save parameter vector
    #-------------------------------------------------------------------------------
    paramVector = np.concatenate((beta, sigma2))
    for k in np.arange(len(nraneffs)):
        paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

    #-------------------------------------------------------------------------------
    # Work out b values
    #-------------------------------------------------------------------------------
    bvals = DinvIplusZtZD @ Zte

    return(paramVector, bvals, nit, llhcurr)


# ============================================================================
# 
# This below function performs Fisher Scoring for the Linear Mixed Model. It
# is based on the update rule:
#
#     \theta_h = \theta_h + lam*I(\theta_h)^(-1) (dl/d\theta_h)
#
# Where \theta_h is the vector (beta, sigma2, vech(D1),...vech(Dr)), lam is a
# scalar stepsize, I(\theta_h) is the Fisher Information matrix of \theta_h 
# and dl/d\theta_h is the derivative of the log likelihood of \theta_h with
# respect to \theta_h.
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X.
#  - `XtY`: X transpose multiplied by Y.
#  - `XtZ`: X transpose multiplied by Z. 
#  - `YtX`: Y transpose multiplied by X.
#  - `YtY`: Y transpose multiplied by Y.
#  - `YtZ`: Y transpose multiplied by Z.
#  - `ZtX`: Z transpose multiplied by X.
#  - `ZtY`: Z transpose multiplied by Y.
#  - `ZtZ`: Z transpose multiplied by Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations.
#  - `init_paramVector`: (Optional) initial estimates of the parameter vector.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `paramVector`: The parameter vector (beta,sigma2,vech(D_1),...,vech(D_r))
#  - `bvals`: Estimates of the random effects vector, b.
#  - `nit`: The number of iterations taken to converge.
#
# ============================================================================
def FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None):
    
    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------
    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[0]

    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of parameters
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()
        
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------
    if init_paramVector is not None:

        # Initial beta and sigma2
        beta = init_paramVector[0:p]
        sigma2 = init_paramVector[p:(p+1)][0,0]
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Initial D (dictionary version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(vech2mat2D(init_paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))

        # Initial D (matrix version)
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                # Add block
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])
        D = D.toarray()

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

    # If we don't have initial values estimate them
    else:

        # Inital beta
        beta = initBeta2D(XtX, XtY)

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Z'e, needed for first iteration
        Zte = ZtY - (ZtX @ beta)

        # Inital D (Dictionary version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
            
        # Matrix version
        D = np.array([])
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                # Add block
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])

    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = 1
    
    # Initial log likelihoods
    llhprev = np.inf
    llhcurr = -np.inf
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k1 and k2
    permdict = dict()
    for k1 in np.arange(len(nraneffs)):
        for k2 in np.arange(len(nraneffs)):
            permdict[str(k1)+str(k2)] = None

    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    # Number of iterations
    nit = 0
    while np.abs(llhprev-llhcurr)>tol:

        # Update nit
        nit = nit+1

        # Change current likelihood to previous
        llhprev = llhcurr

        # --------------------------------------------------------------------------
        # Matrices needed later 
        # --------------------------------------------------------------------------
        # X transpose e and Z transpose e
        Xte = XtY - (XtX @ beta)

        # --------------------------------------------------------------------------
        # Obtain D(I+Z'ZD)^(-1)
        # --------------------------------------------------------------------------
        DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

        # --------------------------------------------------------------------------
        # Derivatives
        # --------------------------------------------------------------------------
        # Derivative wrt beta
        dldB = get_dldB2D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte)

        # Derivative wrt sigma^2
        dldsigma2 = get_dldsigma22D(n, ete, Zte, sigma2, DinvIplusZtZD)
        
        # For each factor, factor k, work out dl/dD_k
        dldDdict = dict()
        for k in np.arange(len(nraneffs)):
            # Store it in the dictionary# Store it in the dictionary
            if ZtZmatdict[k] is None:
                dldDdict[k],ZtZmatdict[k] = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
            else:
                dldDdict[k],_ = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

        # --------------------------------------------------------------------------
        # Covariance of dl/dsigma2
        # --------------------------------------------------------------------------
        covdldsigma2 = n/(2*(sigma2**2))

        # --------------------------------------------------------------------------
        # Construct the Fisher Information matrix
        # --------------------------------------------------------------------------
        FisherInfoMat = np.zeros((tnp,tnp))

        # Add dl/dbeta covariance
        FisherInfoMat[np.ix_(np.arange(p),np.arange(p))] = get_covdldbeta2D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2)

        # Add dl/dsigma2 covariance
        FisherInfoMat[p,p] = covdldsigma2

        # Add dl/dsigma2 dl/dD covariance
        for k in np.arange(len(nraneffs)):

            # Assign to the relevant block
            if ZtZmatdict[k] is None:
                covdldDksigma2,ZtZmatdict[k] = get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, ZtZmat=None)
            else:
                covdldDksigma2,_ = get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, ZtZmat=ZtZmatdict[k])

            # Assign to the relevant block
            FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldDksigma2.reshape(FishIndsDk[k+1]-FishIndsDk[k])
            FisherInfoMat[FishIndsDk[k]:FishIndsDk[k+1],p] = FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]].transpose()

        # Add dl/dD covariance for each pair (k1,k2) of random factors
        for k1 in np.arange(len(nraneffs)):
            for k2 in np.arange(k1+1):

                # Work out the indices of random factor k1 and random factor k2
                IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
                IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

                # Get covariance between D_k1 and D_k2 
                if permdict[str(k1)+str(k2)] is None:
                    FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],permdict[str(k1)+str(k2)] = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict,perm=None)
                else:
                    FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],_ = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict,perm=permdict[str(k1)+str(k2)])

                # Get covariance between D_k1 and D_k2 
                FisherInfoMat[np.ix_(IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(IndsDk1, IndsDk2)].transpose()
        
        # Check Fisher Information matrix is symmetric
        FisherInfoMat = forceSym2D(FisherInfoMat)

        # ----------------------------------------------------------------------
        # Concatenate paramaters and derivatives together
        # ----------------------------------------------------------------------
        paramVector = np.concatenate((beta, np.array([[sigma2]])))
        derivVector = np.concatenate((dldB, dldsigma2))

        for k in np.arange(len(nraneffs)):

            paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))
            derivVector = np.concatenate((derivVector, dupMatTdict[k] @ mat2vec2D(dldDdict[k])))
        
        # ----------------------------------------------------------------------
        # Update step
        # ----------------------------------------------------------------------
        paramVector = paramVector + lam*(np.linalg.solve(FisherInfoMat,derivVector))

        # ----------------------------------------------------------------------
        # Get the new parameters
        # ----------------------------------------------------------------------
        beta = paramVector[0:p]
        sigma2 = paramVector[p:(p+1)][0,0]
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # D (dict version)
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(vech2mat2D(paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
        
        # D (matrix version)
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])

        # --------------------------------------------------------------------------
        # Matrices for next iteration
        # --------------------------------------------------------------------------
        # Recalculate Zte and ete
        Zte = ZtY - (ZtX @ beta)

        # Sum of squared residuals
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Inverse of (I+Z'ZD) multiplied by D
        DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

        # Check sigma2 hasn't hit a boundary
        if sigma2<0:
            sigma2=1e-10

        # ----------------------------------------------------------------------
        # Update the step size and log likelihood
        # ----------------------------------------------------------------------
        llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
        if llhprev>llhcurr:
            lam = lam/2
    
    # --------------------------------------------------------------------------   
    # Get b values     
    # --------------------------------------------------------------------------
    bvals = DinvIplusZtZD @ Zte
    
    return(paramVector, bvals, nit, llhcurr)


# ============================================================================
# 
# This below function performs pesudo Fisher Scoring for the Linear Mixed 
# Model. It is based on the update rule:
#
#     \theta_f = \theta_f + lam*I(\theta_f)^+ (dl/d\theta_f)
#
# Where \theta_f is the vector (beta, sigma2, vec(D1),...vec(Dr)), lam is a
# scalar stepsize, I(\theta_f) is the Fisher Information matrix of \theta_f 
# and dl/d\theta_f is the derivative of the log likelihood of \theta_f with
# respect to \theta_f. 
#
# Note that, as \theta_f is written in terms of 'vec', rather than 'vech',
# (full  vector, 'f', rather than half-vector, 'h'), the information matrix
# will have repeated rows (due to \theta_f having repeated entries). Because
# of this, this method is based on the "pseudo-Inverse" (represented by the 
# + above), hence the name.
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X.
#  - `XtY`: X transpose multiplied by Y.
#  - `XtZ`: X transpose multiplied by Z. 
#  - `YtX`: Y transpose multiplied by X.
#  - `YtY`: Y transpose multiplied by Y.
#  - `YtZ`: Y transpose multiplied by Z.
#  - `ZtX`: Z transpose multiplied by X.
#  - `ZtY`: Z transpose multiplied by Y.
#  - `ZtZ`: Z transpose multiplied by Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations.
#  - `init_paramVector`: (Optional) initial estimates of the parameter vector.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `paramVector`: The parameter vector (beta,sigma2,vech(D_1),...,vech(D_r))
#  - `bvals`: Estimates of the random effects vector, b.
#  - `nit`: The number of iterations taken to converge.
#
# ============================================================================
def pFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None):
    
    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[0]

    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of parameters
    tnp = np.int32(p + 1 + np.sum(nraneffs**2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs**2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()
        
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------
    # Read in initial estimates if we have any
    if init_paramVector is not None:

        # Initial beta and sigma2
        beta = init_paramVector[0:p]
        sigma2 = init_paramVector[p:(p+1)][0,0]
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Initial D (dictionary version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(vec2mat2D(init_paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
            
        # Initial D (matrix version)
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Z'e, needed for first iteration
        Zte = ZtY - (ZtX @ beta)

    # Estimate initial values otherwise
    else:

        # Inital beta
        beta = initBeta2D(XtX, XtY)

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Z'e, needed for first iteration
        Zte = ZtY - (ZtX @ beta)

        # Inital D (Dictionary version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
            
        # Inital D (Matrix version)
        D = np.array([])
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])

    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1) 
    # ------------------------------------------------------------------------------
    DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = 1
    
    # Initial log likelihoods
    llhprev = np.inf
    llhcurr = -np.inf
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k1 and k2
    permdict = dict()
    for k1 in np.arange(len(nraneffs)):
        for k2 in np.arange(len(nraneffs)):
            permdict[str(k1)+str(k2)] = None

    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    nit = 0
    while np.abs(llhprev-llhcurr)>tol:

        # Update number of iterations
        nit = nit+1
        
        # Change current likelihood to previous
        llhprev = llhcurr

        # ------------------------------------------------------------------------
        # Matrices needed later by many calculations:
        # ------------------------------------------------------------------------
        # X transpose e and Z transpose e
        Xte = XtY - (XtX @ beta)

        # ------------------------------------------------------------------------
        # Derivatives
        # ------------------------------------------------------------------------
        # Derivative wrt beta
        dldB = get_dldB2D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte)

        # Derivative wrt sigma^2
        dldsigma2 = get_dldsigma22D(n, ete, Zte, sigma2, DinvIplusZtZD)
        
        # For each factor, factor k, work out dl/dD_k
        dldDdict = dict()
        for k in np.arange(len(nraneffs)):

            # Store it in the dictionary
            if ZtZmatdict[k] is None:
                dldDdict[k],ZtZmatdict[k] = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
            else:
                dldDdict[k],_ = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

        # ------------------------------------------------------------------------
        # Covariance of dl/dsigma2
        # ------------------------------------------------------------------------
        covdldsigma2 = n/(2*(sigma2**2))

        # ------------------------------------------------------------------------
        # Construct the Fisher Information matrix
        # ------------------------------------------------------------------------
        FisherInfoMat = np.zeros((tnp,tnp))

        # Add dl/dbeta covariance
        FisherInfoMat[np.ix_(np.arange(p),np.arange(p))] = get_covdldbeta2D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2)

        # Add dl/dsigma2 covariance
        FisherInfoMat[p,p] = covdldsigma2

        # Add dl/dsigma2 dl/dD covariance
        for k in np.arange(len(nraneffs)):

            # Assign to the relevant block
            if ZtZmatdict[k] is None:
                covdldDksigma2,ZtZmatdict[k] = get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, ZtZmat=None)
            else:
                covdldDksigma2,_ = get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, ZtZmat=ZtZmatdict[k])

            # Assign to the relevant block
            FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldDksigma2.reshape(FishIndsDk[k+1]-FishIndsDk[k])
            FisherInfoMat[FishIndsDk[k]:FishIndsDk[k+1],p] = FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]].transpose()

        # Add dl/dD covariance for each pair (k1,k2) of random factors
        for k1 in np.arange(len(nraneffs)):
            for k2 in np.arange(k1+1):

                # Work out the indices of D_k1 and D_k2
                IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
                IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

                # Get covariance between D_k1 and D_k2 
                if permdict[str(k1)+str(k2)] is None:
                    FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],permdict[str(k1)+str(k2)] = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict,vec=True,perm=None)
                else:
                    FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],_ = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict,vec=True,perm=permdict[str(k1)+str(k2)])

                FisherInfoMat[np.ix_(IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(IndsDk1, IndsDk2)].transpose()

        # Check Fisher Information matrix is symmetric
        FisherInfoMat = forceSym2D(FisherInfoMat)

        # --------------------------------------------------------------------------
        # Concatenate paramaters and derivatives together
        # --------------------------------------------------------------------------
        paramVector = np.concatenate((beta, np.array([[sigma2]])))
        derivVector = np.concatenate((dldB, dldsigma2))

        for k in np.arange(len(nraneffs)):
            paramVector = np.concatenate((paramVector, mat2vec2D(Ddict[k])))
            derivVector = np.concatenate((derivVector, mat2vec2D(dldDdict[k])))

        # --------------------------------------------------------------------------
        # Update step
        # --------------------------------------------------------------------------
        paramVector = paramVector + lam*(np.linalg.solve(FisherInfoMat, derivVector))
        
        # --------------------------------------------------------------------------
        # Get the new parameters
        # --------------------------------------------------------------------------
        beta = paramVector[0:p]
        sigma2 = paramVector[p:(p+1)][0,0]
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # D (dictionary version)
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(vec2mat2D(paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
        
        # D (matrix version)
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                # Add block for each level of each factor
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])

        # --------------------------------------------------------------------------
        # Matrices for next iteration
        # --------------------------------------------------------------------------
        # Recalculate Zte and ete
        Zte = ZtY - (ZtX @ beta)

        # Sum of squared residuals
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Inverse of (I+Z'ZD) multiplied by D
        DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

        # Check sigma2 hasn't hit a boundary
        if sigma2<0:
            sigma2=1e-10

        # --------------------------------------------------------------------------
        # Update the step size and log likelihood
        # --------------------------------------------------------------------------
        llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
        if llhprev>llhcurr:
            lam = lam/2
    
    # ------------------------------------------------------------------------------
    # Record b values
    # ------------------------------------------------------------------------------
    bvals = DinvIplusZtZD @ Zte

    # ------------------------------------------------------------------------------
    # Convert to vech representation.
    # ------------------------------------------------------------------------------
    # Indices for submatrics corresponding to Dks
    IndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # We reshape from (beta,sigma2, vec(D_1),...vec(D_r)) to
    # (beta,sigma2, vech(D_1),...vech(D_r)) for consistency with other functions.
    paramVector_reshaped = np.zeros((np.sum(nraneffs*(nraneffs+1)//2) + p + 1,1))
    paramVector_reshaped[0:(p+1)]=paramVector[0:(p+1)].reshape(p+1,1)

    # Reshape each vec to vech
    for k in np.arange(len(nlevels)):
        paramVector_reshaped[IndsDk[k]:IndsDk[k+1]] = vec2vech2D(paramVector[FishIndsDk[k]:FishIndsDk[k+1]]).reshape(paramVector_reshaped[IndsDk[k]:IndsDk[k+1]].shape)

    # Return results
    return(paramVector_reshaped, bvals, nit, llhcurr)


# ============================================================================
# 
# This below function performs pseudo-Simplified Fisher Scoring for the Linear
# Mixed Model. It is based on the update rules:
#
#               beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)
#
#                    sigma2 = e'V^(-1)e/n
#
#                     for k in {1,...,r};
#     vec(D_k) = \theta_f + lam*I(vec(D_k))^+ (dl/dvec(D_k))
#
# Where:
#  - lam is a scalar stepsize.
#  - I(vec(D_k)) is the Fisher Information matrix of vec(D_k).
#  - dl/dvf(D_k) is the derivative of the log likelihood of vec(D_k) with
#    respect to vec(D_k). 
#  - e is the residual vector (e=Y-X\beta)
#  - V is the matrix (I+ZDZ')
#
# Note that, the updates are written in terms of 'vec', rather than 'vech',
# (full  vector, 'f', rather than half-vector, 'h'), the information matrix
# will have repeated rows (due to vec(D_k) having repeated entries). Because
# of this, this method is based on the "pseudo-Inverse" (represented by the 
# + above), hence the name.
#
# The name "Simplified" here comes from a convention adopted in (Demidenko 
# 2014).
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X.
#  - `XtY`: X transpose multiplied by Y.
#  - `XtZ`: X transpose multiplied by Z. 
#  - `YtX`: Y transpose multiplied by X.
#  - `YtY`: Y transpose multiplied by Y.
#  - `YtZ`: Y transpose multiplied by Z.
#  - `ZtX`: Z transpose multiplied by X.
#  - `ZtY`: Z transpose multiplied by Y.
#  - `ZtZ`: Z transpose multiplied by Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations.
#  - `init_paramVector`: (Optional) initial estimates of the parameter vector.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `paramVector`: The parameter vector (beta,sigma2,vech(D_1),...,vech(D_r))
#  - `bvals`: Estimates of the random effects vector, b.
#  - `nit`: The number of iterations taken to converge.
#
# ============================================================================
def pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None):
    
    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[0]

    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of parameters
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # Work out D indices (there is one block of D per level)
    Dinds = np.zeros(np.sum(nlevels)+1)
    counter = 0
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1
            
    # Last index will be missing so add it
    Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]
    
    # Make sure indices are ints
    Dinds = np.int64(Dinds)

    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()
        
    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------
    # If we have initial estimates use them.
    if init_paramVector is not None:

        # Initial beta and sigma2
        beta = init_paramVector[0:p]
        sigma2 = init_paramVector[p:(p+1)][0,0]
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # D (dict version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(vech2mat2D(init_paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
            
        # D (matrix version)
        D = scipy.sparse.lil_matrix((q,q))
        counter = 0
        for k in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[k]):
                # Add a block for each level of each factor.
                D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1
        D = D.toarray()

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Z'e, needed for first iteration
        Zte = ZtY - (ZtX @ beta)

    # Otherwise work out initial estimates
    else:

        # Inital beta
        beta = initBeta2D(XtX, XtY)

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Z'e; needed for initial D
        Zte = ZtY - (ZtX @ beta)
            
        # Inital D (dict version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
            
        # Inital D (matrix version)
        D = scipy.sparse.lil_matrix((q,q))
        counter = 0
        for k in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[k]):
                # Add a block for each level of each factor.
                D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1
        D = D.toarray()

    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1)
    # ------------------------------------------------------------------------------
    DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + np.array(D) @ ZtZ, D))

    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = 1
    
    # Initial log likelihoods
    llhprev = np.inf
    llhcurr = -np.inf
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k in np.arange(len(nraneffs)):
        permdict[str(k)] = None

    # ------------------------------------------------------------------------------
    # Iteration.
    # ------------------------------------------------------------------------------
    # Number of iterations
    nit = 0
    while np.abs(llhprev-llhcurr)>tol:

        # Change current likelihood to previous
        llhprev = llhcurr

        # Number of iterations
        nit = nit+1

        #---------------------------------------------------------------------------
        # Update beta
        #---------------------------------------------------------------------------
        beta = np.linalg.solve(XtX - XtZ @ DinvIplusZtZD @ ZtX, XtY - XtZ @ DinvIplusZtZD @ ZtY)
        
        #---------------------------------------------------------------------------
        # Update sigma^2
        #---------------------------------------------------------------------------
        sigma2 = 1/n*(ete - Zte.transpose() @ DinvIplusZtZD @ Zte)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary
        
        #---------------------------------------------------------------------------
        # Update D
        #---------------------------------------------------------------------------
        counter = 0
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Work out derivative of D_k
            #-----------------------------------------------------------------------
            if ZtZmatdict[k] is None:
                dldD,ZtZmatdict[k] = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
            else:
                dldD,_ = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

            #-----------------------------------------------------------------------
            # Work out covariance of derivative of D_k
            #-----------------------------------------------------------------------
            if permdict[str(k)] is None:
                covdldDk,permdict[str(k)] = get_covdldDk1Dk22D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=None)
            else:
                covdldDk,_ = get_covdldDk1Dk22D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=permdict[str(k)])

            #-----------------------------------------------------------------------
            # Update step
            #-----------------------------------------------------------------------
            update = lam*np.linalg.solve(forceSym2D(covdldDk),mat2vec2D(dldD))
            update = vec2vech2D(update)
            
            # Update D_k
            Ddict[k] = makeDpd2D(vech2mat2D(mat2vech2D(Ddict[k]) + update))
            
            #-----------------------------------------------------------------------
            # Add D_k back into D
            #-----------------------------------------------------------------------
            for j in np.arange(nlevels[k]):

                D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1
            
            #-----------------------------------------------------------------------
            # Obtain D(I+Z'ZD)^(-1)
            #-----------------------------------------------------------------------
            DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

        # --------------------------------------------------------------------------
        # Matrices for next iteration
        # --------------------------------------------------------------------------
        # Recalculate Zte and ete
        Zte = ZtY - (ZtX @ beta)

        # Sum of squared residuals
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Check sigma2 hasn't hit a boundary
        if sigma2<0:
            sigma2=1e-10

        # --------------------------------------------------------------------------
        # Update step size and likelihood
        # --------------------------------------------------------------------------
        # Update the step size
        llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
        if llhprev>llhcurr:
            lam = lam/2

    # ------------------------------------------------------------------------------
    # Save parameters
    # ------------------------------------------------------------------------------
    paramVector = np.concatenate((beta, sigma2))
    for k in np.arange(len(nraneffs)):
        paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

    # ------------------------------------------------------------------------------
    # Save bvals
    # ------------------------------------------------------------------------------
    bvals = DinvIplusZtZD @ Zte
        
    return(paramVector, bvals, nit, llhcurr)


# ============================================================================
# 
# This below function performs Simplified Fisher Scoring for the Linear Mixed
# Model. It is based on the update rules:
#
#                       beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)
#
#                              sigma2 = e'V^(-1)e/n
#
#                              for k in {1,...,r};
#     vech(D_k) = \theta_f + lam*I(vech(D_k))^(-1) (dl/dvech(D_k))
#
# Where:
#  - lam is a scalar stepsize.
#  - I(vech(D_k)) is the Fisher Information matrix of vech(D_k).
#  - dl/dvh(D_k) is the derivative of the log likelihood of vech(D_k) with
#    respect to vech(D_k). 
#  - e is the residual vector (e=Y-X\beta)
#  - V is the matrix (I+ZDZ')
#
# The name "Simplified" here comes from a convention adopted in (Demidenko 
# 2014).
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - `XtX`: X transpose multiplied by X.
#  - `XtY`: X transpose multiplied by Y.
#  - `XtZ`: X transpose multiplied by Z. 
#  - `YtX`: Y transpose multiplied by X.
#  - `YtY`: Y transpose multiplied by Y.
#  - `YtZ`: Y transpose multiplied by Z.
#  - `ZtX`: Z transpose multiplied by X.
#  - `ZtY`: Z transpose multiplied by Y.
#  - `ZtZ`: Z transpose multiplied by Z.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#  - `tol`: A scalar tolerance value. Iteration stops once successive 
#           log-likelihood values no longer exceed `tol`.
#  - `n`: The number of observations.
#  - `init_paramVector`: (Optional) initial estimates of the parameter vector.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `paramVector`: The parameter vector (beta,sigma2,vech(D_1),...,vech(D_r))
#  - `bvals`: Estimates of the random effects vector, b.
#  - `nit`: The number of iterations taken to converge.
#
# ============================================================================
def SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, init_paramVector=None):
    
    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[0]

    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of parameters
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()

    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------
    # If we have initial estimates use them
    if init_paramVector is not None:

        # Inital beta and sigma2
        beta = init_paramVector[0:p]
        sigma2 = init_paramVector[p:(p+1)][0,0]
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Initial D (dict version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(vech2mat2D(init_paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
            
        # Initial D (matrix version)
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                # Add a block for each level of each factor
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Z'e, needed for first iteration
        Zte = ZtY - (ZtX @ beta)

    # Otherwise work out initial estimates
    else:

        # Inital beta
        beta = initBeta2D(XtX, XtY)

        # Work out e'e
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary

        # Z'e; needed for initial D
        Zte = ZtY - (ZtX @ beta)
            
        # Inital D (dict version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            Ddict[k] = makeDpd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))
            
        # Inital D (matrix version)
        D = np.array([])
        for i in np.arange(len(nraneffs)):
            for j in np.arange(nlevels[i]):
                # Add a block for each level of each factor
                if i == 0 and j == 0:
                    D = Ddict[i]
                else:
                    D = scipy.linalg.block_diag(D, Ddict[i])

    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1) 
    # ------------------------------------------------------------------------------
    DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = 1
    
    # Initial log likelihoods
    llhprev = np.inf
    llhcurr = -np.inf

    # ------------------------------------------------------------------------------
    # D indices
    # ------------------------------------------------------------------------------
    # Work out D indices (there is one block of D per level)
    Dinds = np.zeros(np.sum(nlevels)+1)
    counter = 0
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1
            
    # Last index will be missing so add it
    Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]
    
    # Make sure indices are ints
    Dinds = np.int64(Dinds)
    
    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k in np.arange(len(nraneffs)):
        permdict[str(k)] = None

    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    # Number of iterations
    nit = 0
    while np.abs(llhprev-llhcurr)>tol:

        # Update number of iterations
        nit = nit + 1

        # Change current likelihood to previous
        llhprev = llhcurr
        
        #---------------------------------------------------------------------------
        # Update beta
        #---------------------------------------------------------------------------
        beta = np.linalg.solve(XtX - XtZ @ DinvIplusZtZD @ ZtX, XtY - XtZ @ DinvIplusZtZD @ ZtY)
        
        #---------------------------------------------------------------------------
        # Update sigma^2
        #---------------------------------------------------------------------------
        sigma2 = 1/n*(ete - Zte.transpose() @ DinvIplusZtZD @ Zte)
        sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary
        
        #---------------------------------------------------------------------------
        # Update D_k
        #---------------------------------------------------------------------------
        counter = 0
        for k in np.arange(len(nraneffs)):
            
            #-----------------------------------------------------------------------
            # Work out derivative of dl/dD_k
            #-----------------------------------------------------------------------
            if ZtZmatdict[k] is None:
                dldD,ZtZmatdict[k] = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
            else:
                dldD,_ = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

            #-----------------------------------------------------------------------
            # Work out covariance derivative of dl/dD_k
            #-----------------------------------------------------------------------
            if permdict[str(k)] is None:
                covdldDk,permdict[str(k)] = get_covdldDk1Dk22D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict,perm=None)
            else:
                covdldDk,_ = get_covdldDk1Dk22D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=permdict[str(k)])

            #-----------------------------------------------------------------------
            # Update step
            #-----------------------------------------------------------------------
            update = lam*np.linalg.solve(forceSym2D(covdldDk), dupMatTdict[k] @ mat2vec2D(dldD))
            
            # Update D_k
            Ddict[k] = makeDpd2D(vech2mat2D(mat2vech2D(Ddict[k]) + update))
            
            #-----------------------------------------------------------------------
            # Add D_k back into D
            #-----------------------------------------------------------------------
            for j in np.arange(nlevels[k]):

                D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1
            
            # ----------------------------------------------------------------------
            # Obtain D(I+Z'ZD)^(-1) 
            # ----------------------------------------------------------------------
            DinvIplusZtZD = forceSym2D(np.linalg.solve(np.eye(q) + D @ ZtZ, D)) 

        # --------------------------------------------------------------------------
        # Matrices for next iteration
        # --------------------------------------------------------------------------
        # Recalculate Zte and ete
        Zte = ZtY - (ZtX @ beta)

        # Sum of squared residuals
        ete = ssr2D(YtX, YtY, XtX, beta)

        # Check sigma2 hasn't hit a boundary
        if sigma2<0:
            sigma2=1e-10

        # --------------------------------------------------------------------------
        # Update step size and likelihood
        # --------------------------------------------------------------------------
        llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
        if llhprev>llhcurr:
            lam = lam/2

    # ------------------------------------------------------------------------------
    # Save parameters
    # ------------------------------------------------------------------------------
    paramVector = np.concatenate((beta, sigma2))
    for k in np.arange(len(nraneffs)):
        paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

    # ------------------------------------------------------------------------------
    # Calculate b-values
    # ------------------------------------------------------------------------------
    bvals = DinvIplusZtZD @ Zte
    
    return(paramVector, bvals, nit, llhcurr)
