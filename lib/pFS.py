import time
import os
import numpy as np
import scipy
from lib.tools3d import *
from lib.tools2d import *

def pFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol,n):
  
  t1_total = time.time()
  t1 = time.time()
  
  # Useful scalars
  # ------------------------------------------------------------------------------

  # Number of factors, r
  r = len(nlevels)

  # Number of random effects, q
  q = np.sum(np.dot(nparams,nlevels))

  # Number of fixed effects, p
  p = XtX.shape[1]

  # Number of voxels, nv
  nv = XtY.shape[0]
  
  # Initial estimates
  # ------------------------------------------------------------------------------

  # Inital beta
  beta = initBeta3D(XtX, XtY)
  
  # Work out e'e, X'e and Z'e
  Xte = XtY - (XtX @ beta)
  Zte = ZtY - (ZtX @ beta)
  ete = ssr3D(YtX, YtY, XtX, beta)
  
  # Initial sigma2
  sigma2 = initSigma23D(ete, n)

  Zte = ZtY - (ZtX @ beta) 
  
  # Inital D
  # Dictionary version
  Ddict = dict()
  for k in np.arange(len(nparams)):

    Ddict[k] = makeDnnd3D(initDk3D(k, ZtZ, Zte, sigma2, nlevels, nparams))
  
  # Full version of D
  D = getDfromDict3D(Ddict, nparams, nlevels)
  
  # Duplication matrices
  # ------------------------------------------------------------------------------
  invDupMatdict = dict()
  for i in np.arange(len(nparams)):

    invDupMatdict[i] = np.asarray(invDupMat2D(nparams[i]).todense())
  
  # Index variables
  # ------------------------------------------------------------------------------
  # Work out the total number of paramateres
  tnp = np.int32(p + 1 + np.sum(nparams**2))

  # Indices for submatrics corresponding to Dks
  FishIndsDk = np.int32(np.cumsum(nparams**2) + p + 1)
  FishIndsDk = np.insert(FishIndsDk,0,p+1)

  Zte = ZtY - (ZtX @ beta)
  
  # Inverse of (I+Z'ZD) multiplied by D
  IplusZtZD = np.eye(q) + ZtZ @ D
  DinvIplusZtZD =  forceSym3D(D @ np.linalg.inv(IplusZtZD)) 
  
  # Step size lambda
  lam = np.ones(nv)

  # Initial log likelihoods
  llhprev = -10*np.ones(XtY.shape[0])
  llhcurr = 10*np.ones(XtY.shape[0])
  
  # Vector checking if all voxels converged
  converged_global = np.zeros(nv)
  
  # Vector of saved parameters which have converged
  savedparams = np.zeros((nv, np.int32(np.sum(nparams**2) + p + 1),1))
  
  t2 = time.time()
  print('Setup time: ', t2-t1)
  
  nit=0
  while np.any(np.abs(llhprev-llhcurr)>tol):
    
    t1 = time.time()
    # Change current likelihood to previous
    llhprev = llhcurr
    
    # Work out how many voxels are left
    nv_iter = XtY.shape[0]
    
    # Derivatives
    # ----------------------------------------------------------------------------

    # Derivative wrt beta
    dldB = get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte)  
    
    # Derivative wrt sigma^2
    dldsigma2 = get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD)
    
    # For each factor, factor k, work out dl/dD_k
    dldDdict = dict()
    for k in np.arange(len(nparams)):
      # Store it in the dictionary
      dldDdict[k] = get_dldDk3D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD)
    
    # Covariances
    # ----------------------------------------------------------------------------

    # Construct the Fisher Information matrix
    # ----------------------------------------------------------------------------
    FisherInfoMat = np.zeros((nv_iter,tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2**2))
    
    # Add dl/dsigma2 covariance
    FisherInfoMat[:,p,p] = covdldsigma2
    
    # Add dl/dbeta covariance
    covdldB = get_covdldbeta3D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2)
    FisherInfoMat[np.ix_(np.arange(nv_iter), np.arange(p),np.arange(p))] = covdldB
    
    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nparams)):

      # Get covariance of dldsigma and dldD      
      covdldsigmadD = get_covdldDkdsigma23D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True).reshape(nv_iter,FishIndsDk[k+1]-FishIndsDk[k])
      
      # Assign to the relevant block
      FisherInfoMat[:,p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
      FisherInfoMat[:,FishIndsDk[k]:FishIndsDk[k+1],p:(p+1)] = FisherInfoMat[:,p:(p+1), FishIndsDk[k]:FishIndsDk[k+1]].transpose((0,2,1))
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nparams)):

      for k2 in np.arange(k1+1):

        IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
        IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

        # Get covariance between D_k1 and D_k2 
        covdldDk1dDk2 = get_covdldDk1Dk23D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True)
        
        # Add to FImat
        FisherInfoMat[np.ix_(np.arange(nv_iter), IndsDk1, IndsDk2)] = covdldDk1dDk2
        FisherInfoMat[np.ix_(np.arange(nv_iter), IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(np.arange(nv_iter), IndsDk1, IndsDk2)].transpose((0,2,1))
           
    # Derivative and parameters
    # ----------------------------------------------------------------------------

    # Concatenate paramaters and derivatives together
    # ----------------------------------------------------------------------------
    paramVector = np.concatenate((beta, sigma2.reshape(nv_iter,1,1)),axis=1)
    derivVector = np.concatenate((dldB, dldsigma2.reshape(nv_iter,1,1)),axis=1)

    for k in np.arange(len(nparams)):

      paramVector = np.concatenate((paramVector, mat2vec3D(Ddict[k])),axis=1)
      derivVector = np.concatenate((derivVector, mat2vec3D(dldDdict[k])),axis=1)
    
    
    # Update step
    # ----------------------------------------------------------------------------
    FisherInfoMat = forceSym3D(FisherInfoMat)
    
    paramVector = paramVector + np.einsum('i,ijk->ijk',lam,(np.linalg.inv(FisherInfoMat) @ derivVector))
    
    # Get the new parameters
    beta = paramVector[:,0:p,:]
    sigma2 = paramVector[:,p:(p+1)][:,0,0]
    
    # D as a dictionary
    for k in np.arange(len(nparams)):

      Ddict[k] = makeDnnd3D(vec2mat3D(paramVector[:,FishIndsDk[k]:FishIndsDk[k+1],:]))
      
    # Full version of D
    D = getDfromDict3D(Ddict, nparams, nlevels)
    
    # Sum of squared residuals
    ete = ssr3D(YtX, YtY, XtX, beta)
    Zte = ZtY - (ZtX @ beta)
    
    # Inverse of (I+Z'ZD) multiplied by D
    IplusZtZD = np.eye(q) + (ZtZ @ D)
    DinvIplusZtZD = forceSym3D(D @ np.linalg.inv(IplusZtZD)) 
    
    # Update the step size
    llhcurr = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)
    lam[llhprev>llhcurr] = lam[llhprev>llhcurr]/2
        
    # Work out which voxels converged
    indices_ConAfterIt, indices_notConAfterIt, indices_ConDuringIt, localconverged, localnotconverged = getConvergedIndices(converged_global, (np.abs(llhprev-llhcurr)<tol))
    converged_global[indices_ConDuringIt] = 1

    # Save parameters from this run
    savedparams[indices_ConDuringIt,:,:]=paramVector[localconverged,:,:]

    # Update matrices
    XtY = XtY[localnotconverged, :, :]
    YtX = YtX[localnotconverged, :, :]
    YtY = YtY[localnotconverged, :, :]
    ZtY = ZtY[localnotconverged, :, :]
    YtZ = YtZ[localnotconverged, :, :]
    ete = ete[localnotconverged, :, :]
    DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :, :]

    lam = lam[localnotconverged]
    llhprev = llhprev[localnotconverged]
    llhcurr = llhcurr[localnotconverged]

    beta = beta[localnotconverged, :, :]
    sigma2 = sigma2[localnotconverged]
    D = D[localnotconverged, :, :]

    for k in np.arange(len(nparams)):
      Ddict[k] = Ddict[k][localnotconverged, :, :]
      
    # Matrices needed later by many calculations:
    # ----------------------------------------------------------------------------
    # X transpose e and Z transpose e
    Xte = XtY - (XtX @ beta)
    Zte = ZtY - (ZtX @ beta)
    
    t2 = time.time()
    nit = nit + 1
    #print('Iteration num: ', nit)
    #print('Iteration time: ', t2-t1)
    #print('Num converged:', nv-nv_iter)

  print(nit)  
  #print('Total time taken: ', time.time()-t1_total)
  #print('Estimated NIFTI time (hours): ', 100*100*100/(nv*60*60)*(time.time()-t1_total))
  
  return(savedparams)
