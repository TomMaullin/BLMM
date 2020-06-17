import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import scipy.sparse
import os
np.set_printoptions(threshold=np.inf)
from lib.npMatrix2d import *
from lib.fileio import *

# ============================================================================
# 
# This below function performs pseudo Fisher Scoring for the ADE Linear
# Mixed Model. It is based on the update rules:
#
#               beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)
#
#                    sigma2E = e'V^(-1)e/n
#
#               vec(\sigma2A,\sigma2D) = \theta_f + 
#       lam*I(vec(\sigma2A,\sigma2D))^+ (dl/dvec(\sigma2A,\sigma2D)
#
# Where:
#  - lam is a scalar stepsize.
#  - \sigma2E is the environmental variance in the ADE model
#  - \sigma2A, \sigma2D are the A and D variance in the ADE model divided by
#    the E variance in the ADE model
#  - I(vec(\sigma2A,\sigma2D)) is the Fisher Information matrix of
#    vec(\sigma2A,\sigma2D).
#  - dl/dvec(\sigma2A,\sigma2D) is the derivative of the log likelihood with
#    respect to vec(\sigma2A,\sigma2D). 
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
def pFS_ADE2D(X, Y, nlevels, nraneffs, tol, n, KinshipA, KinshipD, Structmat1stDict, Structmat2nd):
    
    # ------------------------------------------------------------------------------
    # Product matrices of use
    # ------------------------------------------------------------------------------
    XtX = X.transpose() @ X
    XtY = X.transpose() @ Y
    YtX = Y.transpose() @ X
    YtY = Y.transpose() @ Y

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
    # Initial estimates
    # ------------------------------------------------------------------------------
    # If we have initial estimates use them.

    # Inital beta
    beta = initBeta2D(XtX, XtY)

    # Work out e'e
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Initial sigma2
    sigma2 = initSigma22D(ete, n)
    sigma2 = np.maximum(sigma2,1e-20) # Prevent hitting boundary
    
    # Initial zero matrix to hold the matrices Skcov(dl/Dk)Sk'
    FDk = np.zeros((2*r,2*r))

    # Initial zero vector to hold the vectors Sk*dl/dDk
    SkdldDk = np.zeros((2*r,1))

    # Initial residuals
    e = Y - X @ beta

    for k in np.arange(r):

        # Get FDk
        FDk[2*k:(2*k+2),2*k:(2*k+2)]= nlevels[k]*Structmat1stDict[k] @ Structmat1stDict[k].transpose()

        # Get the indices for the factors 
        Ik = fac_indices2D(k, nlevels, nraneffs)

        # Get Ek
        Ek = e[Ik,:].reshape((nlevels[k],nraneffs[k])).transpose()

        # Get Sk*dl/dDk
        SkdldDk[2*k:(2*k+2),:] = Structmat1stDict[k] @ mat2vec2D(nlevels[k]-Ek @ Ek.transpose()/sigma2)

    # Initial vec(sigma^2A/sigma^2E, sigma^2D/sigma^2E)
    dDdAD = 2*Structmat2nd
    vecAE = np.linalg.pinv(dDdAD @ FDk @ dDdAD.transpose()) @ dDdAD @ SkdldDk

    # Inital D (dict version)
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = vecAE[0,0]**2*KinshipA[k] + vecAE[1,0]**2*KinshipD[k]

    # ------------------------------------------------------------------------------
    # Obtain (I+D)^{-1}
    # ------------------------------------------------------------------------------
    invIplusDdict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        invIplusDdict[k] = np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])

    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = 1
    
    # Initial log likelihoods
    llhprev = np.inf
    llhcurr = -np.inf

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
        # Work out X'V^(-1)X and X'V^(-1)Y
        XtinvVX = np.zeros((p,p))
        XtinvVY = np.zeros((p,1))

        # Loop through levels and factors
        for k in np.arange(r):
            for j in np.arange(nlevels[k]):

                # Get the indices for the factors 
                Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

                # Add to sums
                XtinvVX = XtinvVX + X[Ikj,:].transpose() @ invIplusDdict[k] @ X[Ikj,:]
                XtinvVY = XtinvVY + X[Ikj,:].transpose() @ invIplusDdict[k] @ Y[Ikj,:]

        beta = np.linalg.pinv(XtinvVX) @ XtinvVY

        #---------------------------------------------------------------------------
        # Update Residuals, e
        #---------------------------------------------------------------------------
        e = Y - X @ beta
        ete = e.transpose() @ e

        #---------------------------------------------------------------------------
        # Update sigma^2
        #---------------------------------------------------------------------------
        etinvVe = np.zeros((1,1))
        # Loop through levels and factors
        for k in np.arange(r):
            for j in np.arange(nlevels[k]):

                # Get the indices for the factors 
                Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

                # Add to sums
                etinvVe = etinvVe + e[Ikj,:].transpose() @ invIplusDdict[k] @ e[Ikj,:]

        sigma2 = 1/n*etinvVe
        sigma2 = np.maximum(sigma2,1e-20) # Prevent hitting boundary
        #vecAE = np.maximum(vecAE,1e-10)
        
        #---------------------------------------------------------------------------
        # Update D
        #---------------------------------------------------------------------------
        counter = 0

        # Initial zero matrix to hold the matrices Skcov(dl/Dk)Sk'
        FDk = np.zeros((2*r,2*r))

        # Initial zero vector to hold the vectors Sk*dl/dDk
        SkdldDk = np.zeros((2*r,1))
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Work out derivative of D_k
            #-----------------------------------------------------------------------

            # Get the indices for the factors 
            Ik = fac_indices2D(k, nlevels, nraneffs)

            # Get Ek
            Ek = e[Ik,:].reshape((nlevels[k],nraneffs[k])).transpose()
            
            # Calculate S'dl/dDk
            SkdldDk[2*k:(2*k+2),:] =  Structmat1stDict[k] @ mat2vec2D((invIplusDdict[k] @ Ek @ Ek.transpose() @ invIplusDdict[k]/sigma2[0,0])-nlevels[k]*invIplusDdict[k])

            #-----------------------------------------------------------------------
            # Work out covariance of derivative of D_k
            #-----------------------------------------------------------------------

            # Work out (I+Dk)^(-1) \otimes (I+Dk)^(-1)
            kronTerm = np.kron(invIplusDdict[k],invIplusDdict[k])

            # Get FDk
            FDk[2*k:(2*k+2),2*k:(2*k+2)]= nlevels[k]*Structmat1stDict[k] @ kronTerm @ Structmat1stDict[k].transpose()

        #-----------------------------------------------------------------------
        # Update step
        #-----------------------------------------------------------------------
        dDdAD = 2*Structmat2nd*vecAE
        vecAE = vecAE + lam*np.linalg.pinv(dDdAD @ FDk @ dDdAD.transpose()) @ dDdAD @ SkdldDk

        #-----------------------------------------------------------------------
        # Add D_k back into D
        #-----------------------------------------------------------------------
        # Inital D (dict version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            # Construct D using sigma^2A and sigma^2D
            Ddict[k] = vecAE[0,0]**2*KinshipA[k] + vecAE[1,0]**2*KinshipD[k]

        # ------------------------------------------------------------------------------
        # Obtain (I+D)^{-1}
        # ------------------------------------------------------------------------------
        invIplusDdict = dict()
        for k in np.arange(len(nraneffs)):
            # Construct D using sigma^2A and sigma^2D
            invIplusDdict[k] = np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])

        # --------------------------------------------------------------------------
        # Precautionary
        # --------------------------------------------------------------------------
        # Check sigma2 hasn't hit a boundary
        if sigma2<0:
            sigma2=1e-10

        # --------------------------------------------------------------------------
        # Update step size and likelihood
        # --------------------------------------------------------------------------

        # Update e'V^(-1)e
        etinvVe = np.zeros((1,1))
        # Loop through levels and factors
        for k in np.arange(r):
            for j in np.arange(nlevels[k]):

                # Get the indices for the factors 
                Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

                # Add to sums
                etinvVe = etinvVe + e[Ikj,:].transpose() @ invIplusDdict[k] @ e[Ikj,:]

        # Work out log|V| using the fact V is block diagonal
        logdetV = 0
        for k in np.arange(r):
            logdetV = logdetV - nlevels[k]*np.prod(np.linalg.slogdet(invIplusDdict[k]))

        # Work out the log likelihood
        llhcurr = -0.5*(n*np.log(sigma2)+(1/sigma2)*etinvVe + logdetV)

        # Update the step size
        if llhprev>llhcurr:
            lam = lam/2


    # ------------------------------------------------------------------------------
    # Save parameters
    # ------------------------------------------------------------------------------
    paramVector = np.concatenate((beta, np.sqrt(sigma2), vecAE))
        
    return(paramVector, llhcurr)
