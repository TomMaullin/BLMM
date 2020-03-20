import time
import os
import numpy as np
import scipy
from lib.tools3d import *
from lib.tools2d import *

def FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector):
  
  # Useful scalars
  # ------------------------------------------------------------------------------

  # Number of factors, r
  r = len(nlevels)

  # Number of random effects, q
  q = np.sum(np.dot(nparams,nlevels))

  # Number of fixed effects, p
  p = XtX.shape[0]


  # Index variables
  # ------------------------------------------------------------------------------
  # Work out the total number of paramateres
  tnp = np.int32(p + 1 + np.sum(nparams*(nparams+1)/2))

  # Indices for submatrics corresponding to Dks
  FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + p + 1)
  FishIndsDk = np.insert(FishIndsDk,0,p+1)
  #print('inds',FishIndsDk)

  # Duplication matrices
  # ------------------------------------------------------------------------------
  invDupMatdict = dict()
  for i in np.arange(len(nparams)):

    invDupMatdict[i] = invDupMat2D(nparams[i])
    
  # Initial estimates
  # ------------------------------------------------------------------------------

  if init_paramVector is not None:

    #print(init_paramVector.shape)
    beta = init_paramVector[0:p]
    sigma2 = init_paramVector[p:(p+1)][0,0]

    Ddict = dict()
    for k in np.arange(len(nparams)):

      Ddict[k] = makeDnnd2D(vech2mat2D(init_paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
      
    for i in np.arange(len(nparams)):

      for j in np.arange(nlevels[i]):


        if i == 0 and j == 0:

          D = Ddict[i]

        else:

          D = scipy.linalg.block_diag(D, Ddict[i])

  else:

    # Inital beta
    beta = initBeta2D(XtX, XtY)

    # Work out e'e
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Initial sigma2
    sigma2 = initSigma22D(ete, n)

    Zte = ZtY - (ZtX @ beta)

    # Inital D
    # Dictionary version
    Ddict = dict()
    for k in np.arange(len(nparams)):

      Ddict[k] = makeDnnd2D(initDk2D(k, nlevels[k], ZtZ, Zte, sigma2, nparams, nlevels,invDupMatdict))
      
    # Matrix version
    D = np.array([])
    for i in np.arange(len(nparams)):

      for j in np.arange(nlevels[i]):

        if i == 0 and j == 0:

          D = Ddict[i]

        else:

          D = scipy.linalg.block_diag(D, Ddict[i])

  Zte = ZtY - (ZtX @ beta)

  # Inverse of (I+Z'ZD) multiplied by DIplusDZtZ 
  IplusZtZD = np.eye(q) + ZtZ @ D
  DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))

  # Step size lambda
  lam = 1
  
  # Initial log likelihoods
  llhprev = np.inf
  llhcurr = -np.inf
  
  counter = 0
  while np.abs(llhprev-llhcurr)>tol:
    
    #print('nit', counter)
    counter = counter+1
    
    # Change current likelihood to previous
    llhprev = llhcurr

    # Matrices needed later by many calculations:
    # ----------------------------------------------------------------------------
    # X transpose e and Z transpose e
    Xte = XtY - (XtX @ beta)
    Zte = ZtY - (ZtX @ beta)

    # Inverse of (I+Z'ZD) multiplied by D
    IplusZtZD = np.eye(q) + (ZtZ @ D)
    DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))

    # Sum of squared residuals
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Derivatives
    # ----------------------------------------------------------------------------

    # Derivative wrt beta
    dldB = get_dldB2D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte)

    # Derivative wrt sigma^2
    dldsigma2 = get_dldsigma22D(n, ete, Zte, sigma2, DinvIplusZtZD)
    
    # For each factor, factor k, work out dl/dD_k
    dldDdict = dict()
    for k in np.arange(len(nparams)):
      # Store it in the dictionary
      dldDdict[k] = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD)

    # Covariances
    # ----------------------------------------------------------------------------

    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2**2))

    # Construct the Fisher Information matrix
    # ----------------------------------------------------------------------------
    FisherInfoMat = np.zeros((tnp,tnp))

    # Add dl/dbeta covariance
    FisherInfoMat[np.ix_(np.arange(p),np.arange(p))] = get_covdldbeta2D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2)

    # Add dl/dsigma2 covariance
    FisherInfoMat[p,p] = covdldsigma2

    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nparams)):

      # Assign to the relevant block
      FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]] = get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict).reshape(FishIndsDk[k+1]-FishIndsDk[k])
      FisherInfoMat[FishIndsDk[k]:FishIndsDk[k+1],p] = FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]].transpose()

    # Add dl/dD covariance
    for k1 in np.arange(len(nparams)):

      for k2 in np.arange(k1+1):

        IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
        IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

        # Get covariance between D_k1 and D_k2 
        FisherInfoMat[np.ix_(IndsDk1, IndsDk2)] = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict)
        FisherInfoMat[np.ix_(IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(IndsDk1, IndsDk2)].transpose()

    paramVector = np.concatenate((beta, np.array([[sigma2]])))
    derivVector = np.concatenate((dldB, dldsigma2))

    for k in np.arange(len(nparams)):

      paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))
      derivVector = np.concatenate((derivVector, mat2vech2D(dldDdict[k])))

    FisherInfoMat = forceSym2D(FisherInfoMat)

    paramVector = paramVector + lam*(np.linalg.inv(FisherInfoMat) @ derivVector)
    
    if sigma2<0:

      sigspos[z]=1
      sigma2 = np.maximum(sigma2,1e-6)

    #print(paramVector)
    beta = paramVector[0:p]
    sigma2 = paramVector[p:(p+1)][0,0]

    for k in np.arange(len(nparams)):

      Ddict[k] = makeDnnd2D(vech2mat2D(paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
      
    for i in np.arange(len(nparams)):

      for j in np.arange(nlevels[i]):


        if i == 0 and j == 0:

          D = Ddict[i]

        else:

          D = scipy.linalg.block_diag(D, Ddict[i])

    # Update the step size
    llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
    if llhprev>llhcurr:
      lam = lam/2
      
  bvals = DinvIplusZtZD @ Zte
  
  return(paramVector, bvals)
