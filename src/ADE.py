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
def pFS_ADE2D(X, Y, nlevels, nraneffs, tol, n, KinshipA, KinshipD, Structmat1stDict, Structmat2nd, reml=False):
    
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

    #vecAE[1,0]=0

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
    # Precalculated Kronecker sums
    # ------------------------------------------------------------------------------

    XkXdict = dict()
    XkYdict = dict()

    # Loop through levels and factors
    for k in np.arange(r):

        # Get qk
        qk = nraneffs[k]

        # Sum XkX
        XkXdict[k] = np.zeros((p**2,qk**2))

        # Sum XkY
        XkYdict[k] = np.zeros((p,qk**2))

        for j in np.arange(nlevels[k]):

            # Indices for level j of factor k
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Add to running sum
            XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

            # Add to running sum
            XkYdict[k] = XkYdict[k] + np.kron(Y[Ikj,:].transpose(),X[Ikj,:].transpose())

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

        # Maximum number of iterations
        # if nit>100:
        #     print('nit lim')
        #     break

        #---------------------------------------------------------------------------
        # Update beta
        #---------------------------------------------------------------------------
        # Work out X'V^(-1)X and X'V^(-1)Y
        XtinvVX = np.zeros((p,p))
        XtinvVY = np.zeros((p,1))

        # XtinvVX2 = np.zeros((p,p))
        # XtinvVY2 = np.zeros((p,1))


        # Loop through levels and factors
        for k in np.arange(r):
            # for j in np.arange(nlevels[k]):

            #     # Get the indices for the factors 
            #     Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            #     # Add to sums
            #     XtinvVX = XtinvVX + forceSym2D(X[Ikj,:].transpose() @ invIplusDdict[k] @ X[Ikj,:])
            #     XtinvVY = XtinvVY + X[Ikj,:].transpose() @ invIplusDdict[k] @ Y[Ikj,:]

            XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(invIplusDdict[k]),shape=np.array([p,p]))
            XtinvVY = XtinvVY + vec2mat2D(XkYdict[k] @ mat2vec2D(invIplusDdict[k]),shape=np.array([p,1]))

        # print(np.allclose(XtinvVY, XtinvVY2))
        # print(np.allclose(XtinvVX, XtinvVX2))

        beta = np.linalg.solve(forceSym2D(XtinvVX), XtinvVY)
        # beta2 = np.linalg.pinv(XtinvVX2) @ XtinvVY2

        # print(np.allclose(beta,beta2))

        #---------------------------------------------------------------------------
        # Update Residuals, e
        #---------------------------------------------------------------------------
        e = Y - X @ beta
        ete = e.transpose() @ e

        # e2 = Y - X @ beta2
        # ete2 = e2.transpose() @ e2

        # print('etes: ', np.allclose(ete2,ete))

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

        if not reml:
            sigma2 = 1/n*etinvVe
        else:
            sigma2 = 1/(n-p)*etinvVe
        sigma2 = np.maximum(sigma2,1e-20) # Prevent hitting boundary

        # Initial zero matrix to hold F
        F = np.zeros((2,2))

        # Initial zero vector to hold the vectors Sk*dl/dDk
        S = np.zeros((2,1))

        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Work out derivative of D_k
            #-----------------------------------------------------------------------

            # Get the indices for the factors
            Ik = fac_indices2D(k, nlevels, nraneffs)

            # Get Ek
            Ek = e[Ik,:].reshape((nlevels[k],nraneffs[k])).transpose()
            
            # # Calculate S'dl/dDk
            # SkdldDk2[2*k:(2*k+2),:] =  Structmat1stDict[k] @ mat2vec2D((invIplusDdict[k] @ Ek2 @ Ek2.transpose() @ invIplusDdict[k]/sigma22[0,0])-nlevels[k]*invIplusDdict[k])

            if not reml:
                S = S + Structmat1stDict[k] @ mat2vec2D(forceSym2D((invIplusDdict[k] @ Ek @ Ek.transpose() @ invIplusDdict[k]/sigma2[0,0])-nlevels[k]*invIplusDdict[k]))
            else:
                CurrentS = mat2vec2D(forceSym2D((invIplusDdict[k] @ Ek @ Ek.transpose() @ invIplusDdict[k]/sigma2[0,0])-nlevels[k]*invIplusDdict[k]))
                CurrentS =  CurrentS + np.kron(invIplusDdict[k],invIplusDdict[k]) @ XkXdict[k].transpose() @ mat2vec2D(np.linalg.pinv(XtinvVX))
                S = S + Structmat1stDict[k] @ CurrentS

            #-----------------------------------------------------------------------
            # Work out covariance of derivative of D_k
            #-----------------------------------------------------------------------

            # Work out (I+Dk)^(-1) \otimes (I+Dk)^(-1)
            kronTerm = np.kron(invIplusDdict[k],invIplusDdict[k])

            # Get F for this term
            F = F + forceSym2D(nlevels[k]*Structmat1stDict[k] @ kronTerm @ Structmat1stDict[k].transpose())

        #-----------------------------------------------------------------------
        # Update step
        #-----------------------------------------------------------------------

        vecAE = vecAE + forceSym2D(0.5*lam*np.linalg.pinv(forceSym2D(F)*(vecAE @ vecAE.transpose()))) @ (vecAE*S)
        
        #-----------------------------------------------------------------------
        # Add D_k back into D
        #-----------------------------------------------------------------------
        # Inital D (dict version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            # Construct D using sigma^2A and sigma^2D
            Ddict[k] = forceSym2D(vecAE[0,0]**2*KinshipA[k] + vecAE[1,0]**2*KinshipD[k])

        # ------------------------------------------------------------------------------
        # Obtain (I+D)^{-1}
        # ------------------------------------------------------------------------------
        invIplusDdict = dict()
        for k in np.arange(len(nraneffs)):
            # Construct D using sigma^2A and sigma^2D
            invIplusDdict[k] = forceSym2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]))

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

        if reml:
            logdet = np.linalg.slogdet(XtinvVX)
            llhcurr = llhcurr - 0.5*logdet[0]*logdet[1] + 0.5*p*np.log(sigma2)

        # Update the step size
        if llhprev>llhcurr:
            lam = lam/2


    # ------------------------------------------------------------------------------
    # Save parameters
    # ------------------------------------------------------------------------------
    paramVector = np.concatenate((beta, np.sqrt(sigma2), vecAE))

    print('Nit: ', nit)
        
    return(paramVector, llhcurr)



def get_swdf_ADE_T2D(L, paramVec, X, nlevels, nraneffs, KinshipA, KinshipC, Structmat1stDict): 

    # Work out n and p
    n = X.shape[0]
    p = X.shape[1]

    # Work out beta, sigma2 and the vector of variance components
    beta = paramVec[0:p,:]
    sigma2 = paramVec[p,0]**2
    vecAE = paramVec[(p+1):,:]

    # Get D in dictionary form
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = vecAE[0,0]**2*KinshipA[k] + vecAE[1,0]**2*KinshipC[k]

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB_ADE_2D(L, X, Ddict, sigma2, nlevels, nraneffs)

    # Get derivative of S^2
    dS2 = get_dS2_ADE_2D(L, X, Ddict, vecAE, sigma2, nlevels, nraneffs, Structmat1stDict)

    # Get Fisher information matrix
    InfoMat = get_InfoMat_ADE_2D(Ddict, vecAE, sigma2, n, nlevels, nraneffs, Structmat1stDict)

    # Calculate df estimator
    df = 2*(S2**2)/(dS2.transpose() @ np.linalg.solve(InfoMat, dS2))

    # Return df
    return(df)


def get_varLB_ADE_2D(L, X, Ddict, sigma2, nlevels, nraneffs):

    # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
    varLB = L @ get_covB_ADE_2D(X, Ddict, sigma2, nlevels, nraneffs) @ L.transpose()

    # Return result
    return(varLB)


def get_covB_ADE_2D(X, Ddict, sigma2, nlevels, nraneffs):

    # Work out p and r
    p = X.shape[1]
    r = len(nlevels)

    # Work out sum over j of X_(k,j) kron X_(k,j), for each k
    XkXdict = dict()

    # Loop through levels and factors
    for k in np.arange(r):

        # Get qk
        qk = nraneffs[k]

        # Sum XkX
        XkXdict[k] = np.zeros((p**2,qk**2))

        for j in np.arange(nlevels[k]):

            # Indices for level j of factor k
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Add to running sum
            XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

    # Work out X'V^(-1)X as matrix reshape of (sum over k of ((sum_j X_(k,j) kron X_(k,j))vec(D_k)))
    XtinvVX = np.zeros((p,p))

    # Loop through levels and factors
    for k in np.arange(r):

        XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])),shape=np.array([p,p]))

    # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
    covB = np.linalg.pinv(XtinvVX)

    # Calculate sigma^2(X'V^{-1}X)^(-1)
    covB = sigma2*covB

    # Return result
    return(covB)

def get_dS2_ADE_2D(L, X, Ddict, vecAE, sigma2, nlevels, nraneffs, Structmat1stDict):

    # Work out r
    r = len(nlevels)

    # Work out p
    p = X.shape[1]

    # Work out sum over j of X_(k,j) kron X_(k,j), for each k
    XkXdict = dict()

    # Loop through levels and factors
    for k in np.arange(r):

        # Get qk
        qk = nraneffs[k]

        # Sum XkX
        XkXdict[k] = np.zeros((p**2,qk**2))

        for j in np.arange(nlevels[k]):

            # Indices for level j of factor k
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Add to running sum
            XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

    # Work out X'V^(-1)X as matrix reshape of (sum over k of ((sum_j X_(k,j) kron X_(k,j))vec(D_k)))
    XtinvVX = np.zeros((p,p))

    # Loop through levels and factors
    for k in np.arange(r):

        XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])),shape=np.array([p,p]))

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2 = np.zeros((3,1))

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.pinv(XtinvVX) @ L.transpose()

    # Add to dS2
    dS2[0:1,0] = dS2dsigma2.reshape(dS2[0:1,0].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((nraneffs[k]**2,1))

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]) @ X[Ikj,:]

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.pinv(XtinvVX) @ L.transpose()

            # Sum terms
            dS2dvechDk = dS2dvechDk + mat2vec2D(np.kron(K,K.transpose()))

        # Multiply by sigma^2
        dS2dvechDk = sigma2*dS2dvechDk

        # Add to dS2
        dS2[1:,0:1] = dS2[1:,0:1] + Structmat1stDict[k] @ dS2dvechDk.reshape((nraneffs[k]**2,1))

    # Multiply by 2vecAE elementwise
    dS2[1:,0:1] = 2*vecAE*dS2[1:,0:1]

    return(dS2)


def get_InfoMat_ADE_2D(Ddict, vecAE, sigma2, n, nlevels, nraneffs, Structmat1stDict):

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Index variables
    # ------------------------------------------------------------------------------

    # Initialize FIsher Information matrix
    FisherInfoMat = np.zeros((3,3))

    # Covariance of dl/dsigma2
    C = n/(2*sigma2**2)

    # Add dl/dsigma2 covariance
    FisherInfoMat[0,0] = C

    H = np.zeros((2,1))

    # Get H= cov(dl/sigmaE^2, dl/((sigmaA,sigmaD)/sigmaE))
    for k in np.arange(len(nraneffs)):

        # Get covariance of dldsigma and dldD      
        H = H + Structmat1stDict[k] @ get_covdldDkdsigma2_ADE_2D(k, sigma2, nlevels, nraneffs, Ddict).reshape((nraneffs[k]**2,1))

    # Assign to the relevant block
    FisherInfoMat[1:,0:1] = 2*vecAE*H
    FisherInfoMat[0:1,1:] = FisherInfoMat[1:,0:1].transpose()

    # Initial zero matrix to hold F
    F = np.zeros((2,2))

    for k in np.arange(len(nraneffs)):

        #-----------------------------------------------------------------------
        # Work out covariance of derivative of D_k
        #-----------------------------------------------------------------------

        # Work out (I+Dk)^(-1) \otimes (I+Dk)^(-1)
        kronTerm = np.kron(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]),np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]))

        # Get F for this term
        F = F + forceSym2D(nlevels[k]*Structmat1stDict[k] @ kronTerm @ Structmat1stDict[k].transpose())

    # Multiply by 2vecAE elementwise on both sides
    F = 2*forceSym2D(F)*(vecAE @ vecAE.transpose())

    # Assign to the relevant block
    FisherInfoMat[1:, 1:] = F

    # Return result
    return(FisherInfoMat)


def get_covdldDkdsigma2_ADE_2D(k, sigma2, nlevels, nraneffs, Ddict):

    # Get the indices for the factors 
    Ik = fac_indices2D(k, nlevels, nraneffs)

    # Work out lk
    lk = nlevels[k]

    # Work out block size
    qk = nraneffs[k]

    # Obtain sum of Rk = lk*(I+Dk)^(-1)
    RkSum = lk*np.linalg.pinv(np.eye(qk)+Ddict[k])

    # save and return
    covdldDdldsigma2 = 1/(2*sigma2) * mat2vec2D(RkSum)  

    return(covdldDdldsigma2)



def get_T_ADE_2D(L, X, paramVec, KinshipA, KinshipC, nlevels, nraneffs):

    # Work out n and p
    n = X.shape[0]
    p = X.shape[1]

    # Work out beta, sigma2 and the vector of variance components
    beta = paramVec[0:p,:]
    sigma2 = paramVec[p,0]**2
    vecAE = paramVec[(p+1):,:]

    # Get D in dictionary form
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = vecAE[0,0]**2*KinshipA[k] + vecAE[1,0]**2*KinshipC[k]
    
    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Work out Lbeta
    LB = L @ beta

    # Work out se(T)
    varLB = get_varLB_ADE_2D(L, X, Ddict, sigma2, nlevels, nraneffs)

    # Work out T
    T = LB/np.sqrt(varLB)

    # Return T
    return(T)