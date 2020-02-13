import numpy as np
import scipy.sparse
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack
from lib.tools2d import faclev_indices2D, invDupMat2D, mat2vech2D, get_mapping2D, mapping2D
from lib.PLS import PLS2D_getSigma2, PLS2D_getD, PLS2D_getBeta, PLS2D
from lib.tools3d import kron3D, mat2vech3D, get_covdldDkdsigma23D, get_covdldDk1Dk23D, forceSym3D
import numdifftools as nd
import sys

# ============================================================================
#
# The function below is a wrapper function for all Sattherthwaite degrees of 
# freedom estimation used by BLMM.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `YtX`: Y transpose multiplied by X (Y'X in the above notation).
# - `YtY`: Y transpose multiplied by Y (Y'Y in the above notation).
# - `XtX`: X transpose multiplied by X (X'X in the above notation).
# - `beta`: An estimate of the parameter vector (\beta in the above notation).
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `ete`: The sum of square residuals (e'e in the above notation).
#
# ============================================================================
def SattherthwaiteDoF(statType,estType,D,sigma2,L,ZtX,ZtY,XtX,ZtZ,XtY,YtX,YtZ,XtZ,YtY,n,nlevels,nparams,theta):

    # T contrast
    if statType=='T':

        # Use lmerTest method
        if estType=='lmerTest':

            # Get estimated degrees of freedom
            df = SW_lmerTest(theta,L,nlevels,nparams,ZtX,ZtY,XtX,ZtZ,XtY,YtX,YtZ,XtZ,YtY,n)

        # Use BLMM method
        else:
            
            # Get estimated degrees of freedom
            df = SW_BLMM(D, sigma2, L, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, nlevels, nparams)
            
    else:

        pass

    return(df)

def SW_lmerTest(theta3D,L,nlevels,nparams,ZtX,ZtY,XtX,ZtZ,XtY,YtX,YtZ,XtZ,YtY,n):# TODO inputs

    #================================================================================
    # Initial theta
    #================================================================================
    theta0 = np.array([])
    r = np.amax(nlevels.shape)
    for i in np.arange(r):
      theta0 = np.hstack((theta0, mat2vech2D(np.eye(nparams[i])).reshape(np.int64(nparams[i]*(nparams[i]+1)/2))))
  
    #================================================================================
    # Sparse Permutation, P
    #================================================================================
    tinds,rinds,cinds=get_mapping2D(nlevels, nparams)
    Lam=mapping2D(np.random.randn(theta0.shape[0]),tinds,rinds,cinds)

    # Obtain Lambda'Z'ZLambda
    LamtZtZLam = spmatrix.trans(Lam)*cvxopt.sparse(matrix(ZtZ[0,:,:]))*Lam

    # Obtaining permutation for PLS
    cholmod.options['supernodal']=2
    P=amd.order(LamtZtZLam)

    # Identity
    I = spmatrix(1.0, range(Lam.size[0]), range(Lam.size[0]))

    # These are not spatially varying
    XtX_current = cvxopt.matrix(XtX[0,:,:])
    XtZ_current = cvxopt.matrix(XtZ[0,:,:])
    ZtX_current = cvxopt.matrix(ZtX[0,:,:])
    ZtZ_current = cvxopt.sparse(cvxopt.matrix(ZtZ[0,:,:]))

    df = np.zeros(YtY.shape[0])

    # Get the sigma^2 and D estimates.
    for i in np.arange(theta3D.shape[0]):

        # Get current theta
        theta = theta3D[i,:]

        # Convert product matrices to CVXopt form
        XtY_current = cvxopt.matrix(XtY[i,:,:])
        YtX_current = cvxopt.matrix(YtX[i,:,:])
        YtY_current = cvxopt.matrix(YtY[i,:,:])
        YtZ_current = cvxopt.matrix(YtZ[i,:,:])
        ZtY_current = cvxopt.matrix(ZtY[i,:,:])

        # # Obtain beta estimate
        # beta = np.array(PLS2D_getBeta(theta, ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, YtZ_current, XtZ_current, YtY_current, n, P, tinds, rinds, cinds))


        #NTS CURRENTLY FOR SPARSE CHOL, NOT (\sigma,SPCHOL(D))
        #ALSO MIGHT HAVE PROBLEMS WITH CVXOPT CONVERSION

        # Convert to gamma form
        gamma = theta2gamma(theta, ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, YtZ_current, XtZ_current, YtY_current, n, P, I, tinds, rinds, cinds)

        # How to get the log likelihood from gammma
        def llhgamma(g, ZtX=ZtX_current, ZtY=ZtY_current, XtX=XtX_current, ZtZ=ZtZ_current, XtY=XtY_current, 
                   YtX=YtX_current, YtZ=YtZ_current, XtZ=XtZ_current, YtY=YtY_current, n=n, P=P, I=I, 
                   tinds=tinds, rinds=rinds, cinds=cinds): 

            t = gamma2theta(g)
            return PLS2D(t, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds)

        # print('gamma')
        # print(gamma)
        # print('theta')
        # print(gamma2theta(gamma))

        # Estimate hessian
        H = nd.Hessian(llhgamma)(gamma)

        # print('H shape')
        # print(H.shape)
        # print('H')
        # print(H)

        # How to get S^2 from gamma
        def S2gamma(g, L=L, ZtX=ZtX_current, ZtY=ZtY_current, XtX=XtX_current, ZtZ=ZtZ_current, XtY=XtY_current, 
                  YtX=YtX_current, YtZ=YtZ_current, XtZ=XtZ_current, YtY=YtY_current, n=n, P=P, I=I,
                  tinds=tinds, rinds=rinds, cinds=cinds):
            return(S2_gamma(g, L, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds))

        # Estimate Jacobian
        J = nd.Jacobian(S2gamma)(gamma)

        # print('J shape')
        # print(J.shape)

        # Calulcate S^2
        S2 = S2_gamma(gamma, L, ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, 
                      YtZ_current, XtZ_current, YtY_current, n, P, I, tinds, rinds, cinds)


        if i==10:

            print('numerator')
            print(2*(S2**2))
            print('denominator')
            print((J @ np.linalg.pinv(H) @ J.transpose()))

        # Calculate the degrees of freedom
        df[i] = 2*(S2**2)/(J @ np.linalg.pinv(H) @ J.transpose())

    return(df)


def SW_BLMM(D, sigma2, L, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, nlevels, nparams): 

    print('SW_BLMM running')

    # Get S^2 of eta
    S2 = S2_eta(D, sigma2, L, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY)
    
    # Get derivative of S^2 with respect to gamma evaluated at eta.
    dS2 = dS2deta(nparams, nlevels, L, XtX, XtZ, ZtZ, ZtX, D, sigma2)

    # Get Fisher information matrix
    InfoMat = InfoMat_eta(D, sigma2, n, nlevels, nparams, ZtZ)#...

    print("S2 shape: ", S2.shape)
    print("I shape: ",InfoMat.shape)
    print("dS2 shape: ", dS2.shape)

    # Calculate df estimator
    df = 2*(S2**2)/(dS2.transpose(0,2,1) @ np.linalg.inv(InfoMat) @ dS2)

    print('df shape ', df.shape)


    print('numerator')
    print(2*(S2[10,:,:]**2))
    print('denominator')
    print((dS2.transpose(0,2,1) @ np.linalg.inv(InfoMat) @ dS2)[10,:,:])

    # Return df
    return(df)

# Parameter formulations used by different softwares:
#
# theta:
#   - used by lmer
#   - has the form (vech(spChol(sigma^2*D_1)),...,vech(spChol(sigma^2*D_r)))
# gamma:
#   - used by lmerTest
#   - has the form (sigma, vech(spChol(D_1)),... vech(spChol(D_r)))
# eta:
#   - used by PLS
#   - has the form (sigma^2, vech(D_1),...vech(D_r))



### NTS CHANGE THETA TO GAMMA TO SHOW DIFF BETWEEN LMER AND LMERTEST THETA
def gamma2theta(gamma):

    # Obtain sigma
    sigma = gamma[0]

    # Multiply remainder of gamma by sigma
    theta = gamma[1:]*sigma

    # Return theta
    return(theta)


def theta2gamma(theta, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds):

    # Obtain sigma^2 estimate
    sigma2 = PLS2D_getSigma2(theta, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds)
    
    # We need sigma
    sigma = np.sqrt(sigma2)

    # Obtain gamma
    gamma = np.concatenate((sigma, theta/sigma),axis=None)

    # Return gamma
    return(gamma)


def S2_gamma(gamma, L, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds):

    # Get theta from gamma
    theta = gamma2theta(gamma)

    # Obtain sigma^2 estimate
    sigma2 = np.array(PLS2D_getSigma2(theta, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds))[0,0]

    # print('theta')
    # print(theta)
    # print('theta shape')
    # print(theta.shape)
    # print('sigma2')
    # print(sigma2)
    # print('sigma2 shape')
    # print(sigma2.shape)
    # tmp = np.array(matrix(PLS2D_getD(theta, tinds, rinds, cinds, sigma2)))
    # print('get D result')
    # print(tmp)
    # print('get D result (type)')
    # print(type(tmp))

    # Obtain D estimate
    D = np.array(matrix(PLS2D_getD(theta, tinds, rinds, cinds, sigma2)))

    # print('in multiplication')
    # print(D.shape)
    # print(np.array(matrix(I)).shape)
    # print(np.array(XtX).shape)
    # print(np.array(ZtX).shape)
    # print(np.array(XtZ).shape)
    # print(np.array(matrix(ZtZ)).shape)

    # Calculate X'V^{-1}X=X'(I+ZDZ')^{-1}X=X'X-X'Z(I+DZ'Z)^{-1}DZ'X
    XtiVX = np.array(XtX) - np.array(XtZ) @ np.linalg.inv(np.array(matrix(I)) + D @ np.array(matrix(ZtZ))) @ D @ np.array(ZtX)

    # Calculate S^2 = sigma^2L(X'V^{-1}X)L'
    S2 = sigma2*L @ np.linalg.inv(XtiVX) @ L.transpose()

    if np.random.uniform(0,1,1)<0.01:

        print('S2')
        print(S2)
        print('XtiVX')
        print(XtiVX)
        print('theta')
        print(theta)
        print('XtX')
        print(XtX)
        np.set_printoptions(threshold=sys.maxsize)
        print('D')
        print(D)
        print('Sigma2')
        print(sigma2)

    return(S2)

def S2_eta(D, sigma2, L, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY):

    print('S2_eta running')

    # Calculate X'V^{-1}X=X'(I+ZDZ')^{-1}X=X'X-X'Z(I+DZ'Z)^{-1}DZ'X
    XtiVX = XtX - XtZ @ np.linalg.inv(np.eye(D.shape[1]) + D @ ZtZ) @ D @ ZtX

    # Calculate S^2 = sigma^2L(X'V^{-1}X)L'
    S2 = np.einsum('i,ijk->ijk',sigma2,(L @ np.linalg.inv(XtiVX) @ L.transpose()))

    return(S2)


def dS2deta(nparams, nlevels, L, XtX, XtZ, ZtZ, ZtX, D, sigma2):

    print('dS2deta running')

    # Number of voxels
    nv = D.shape[0]

    # Calculate X'V^{-1}X=X'(I+ZDZ')^{-1}X=X'X-X'Z(I+DZ'Z)^{-1}DZ'X
    XtiVX = XtX - XtZ @ np.linalg.inv(np.eye(D.shape[1]) + D @ ZtZ) @ D @ ZtX

    # New empty array for differentiating S^2 wrt gamma.
    dS2deta = np.zeros((nv, 1+np.int32(np.sum(nparams*(nparams+1)/2)),1))

    # Work out indices for each start of each component of vector 
    # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
    DerivInds = np.int32(np.cumsum(nparams*(nparams+1)/2) + 1)
    DerivInds = np.insert(DerivInds,0,1)

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.inv(XtiVX) @ L.transpose()

    # Add to dS2deta
    dS2deta[:,0:1] = dS2dsigma2.reshape(dS2deta[:,0:1].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nparams)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((np.int32(nparams[k]*(nparams[k]+1)/2),1))#...

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nparams)
                    
            # Work out Z_(k,j)'Z
            ZkjtZ = ZtZ[:,Ikj,:]

            # Work out Z_(k,j)'X
            ZkjtX = ZtX[:,Ikj,:]

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = ZkjtX - ZkjtZ @ np.linalg.inv(np.eye(D.shape[1]) + D @ ZtZ) @ D @ ZtX

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.inv(XtiVX) @ L.transpose()
            
            # Sum terms
            dS2dvechDk = dS2dvechDk + mat2vech3D(kron3D(K,K.transpose(0,2,1)))

        # Multiply by sigma^2
        dS2dvechDk = np.einsum('i,ijk->ijk',sigma2,dS2dvechDk)

        # Add to dS2deta
        dS2deta[:,DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2deta[:,DerivInds[k]:DerivInds[k+1]].shape)

    return(dS2deta)

def InfoMat_eta(D, sigma2, n, nlevels, nparams, ZtZ):


    print('InfoMat_eta running')

    # Number of random effects, q
    q = np.sum(np.dot(nparams,nlevels))

    # Number of voxels 
    nv = sigma2.shape[0]

    # Duplication matrices
    # ------------------------------------------------------------------------------
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):

        invDupMatdict[i] = np.asarray(invDupMat2D(nparams[i]).todense())

    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(1 + np.sum(nparams*(nparams+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + 1)
    FishIndsDk = np.insert(FishIndsDk,0,1)

    # Inverse of (I+Z'ZD) multiplied by D
    IplusZtZD = np.eye(q) + ZtZ @ D
    DinvIplusZtZD =  forceSym3D(D @ np.linalg.inv(IplusZtZD)) 

    # Initialize FIsher Information matrix
    FisherInfoMat = np.zeros((nv,tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2**2))
    
    # Add dl/dsigma2 covariance
    FisherInfoMat[:,0,0] = covdldsigma2

    
    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nparams)):

        # Get covariance of dldsigma and dldD      
        covdldsigmadD = get_covdldDkdsigma23D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict).reshape(nv,FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FisherInfoMat[:,0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
        FisherInfoMat[:,FishIndsDk[k]:FishIndsDk[k+1],0:1] = FisherInfoMat[:,0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose((0,2,1))
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nparams)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk23D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict)

            # Add to FImat
            FisherInfoMat[np.ix_(np.arange(nv), IndsDk1, IndsDk2)] = covdldDk1dDk2
            FisherInfoMat[np.ix_(np.arange(nv), IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(np.arange(nv), IndsDk1, IndsDk2)].transpose((0,2,1))


    # Return result
    return(FisherInfoMat)