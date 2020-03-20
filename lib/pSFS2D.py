import time
import os
import numpy as np
import scipy
from lib.tools3d import *
from lib.tools2d import *

def pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector):
  
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

      Ddict[k] = makeDnnd2D(initDk2D(k, nlevels[k], ZtZ, Zte, sigma2, nparams, nlevels, invDupMatdict))
      
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

  # Work out D indices (there is one block of D per level)
  Dinds = np.zeros(np.sum(nlevels)+1)
  counter = 0
  for k in np.arange(len(nparams)):
    for j in np.arange(nlevels[k]):
      Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nparams)))[k] + nparams[k]*j
      counter = counter + 1
      
  # Last index will be missing so add it
  Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nparams[-1]
  
  # Make sure indices are ints
  Dinds = np.int64(Dinds)
  
  while np.abs(llhprev-llhcurr)>tol:
    
    # Change current likelihood to previous
    llhprev = llhcurr
    
    #---------------------------------------------------------------------------
    # Update beta
    beta = np.linalg.solve(XtX - XtZ @ DinvIplusZtZD @ ZtX, XtY - XtZ @ DinvIplusZtZD @ ZtY)
    
    # Update sigma^2
    ete = ssr2D(YtX, YtY, XtX, beta)
    Zte = ZtY - (ZtX @ beta)
    sigma2 = 1/n*(ete - Zte.transpose() @ DinvIplusZtZD @ Zte)
    
    # Update D_k
    counter = 0
    for k in np.arange(len(nparams)):
      
      # Work out update amount
      update = lam*forceSym2D(np.linalg.inv(get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict,vec=True))) @ mat2vec2D(get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD))
      update = vec2vech2D(update)
      
      # Update D_k
      Ddict[k] = makeDnnd2D(vech2mat2D(mat2vech2D(Ddict[k]) + update))
      
      # Add D_k back into D and recompute DinvIplusZtZD
      for j in np.arange(nlevels[k]):

        D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
        counter = counter + 1
      
      #D = getDfromDict2D(Ddict,nparams,nlevels)
      
      # Inverse of (I+Z'ZD) multiplied by D
      IplusZtZD = np.eye(q) + (ZtZ @ D)
      DinvIplusZtZD = forceSym2D(D @ np.linalg.inv(IplusZtZD)) 

    # Update the step size
    llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
    if llhprev>llhcurr:
      lam = lam/2

  paramVector = np.concatenate((beta, sigma2))
  for k in np.arange(len(nparams)):
    paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

  bvals = DinvIplusZtZD @ Zte
  
  return(paramVector, bvals)
