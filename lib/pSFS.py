import time
import os
import numpy as np
import scipy
from lib.tools3d import *
from lib.tools2d import *

def pSFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol,n):
  
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
  dupInvDupMatdict = dict()
  dupDuptMatdict = dict()
  for i in np.arange(len(nparams)):

    invDupMatdict[i] = np.asarray(invDupMat2D(nparams[i]).todense())
    #dupInvDupMatdict[i] = np.asarray(dupMat(nparams[i]).todense()) @ invDupMatdict[i]
    #dupDuptMatdict[i] = np.asarray(dupMat(nparams[i]).todense()) @ np.asarray(dupMat(nparams[i]).todense()).transpose()
  
  # Index variables
  # ------------------------------------------------------------------------------
  # Work out the total number of paramateres
  tnp = np.int32(p + 1 + np.sum(nparams*(nparams+1)/2))

  # Indices for submatrics corresponding to Dks
  FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + p + 1)
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
  savedparams = np.zeros((nv, np.int32(np.sum(nparams*(nparams+1)/2) + p + 1),1))
  
  
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
  
  t2 = time.time()
  #print('Setup time: ', t2-t1)
  
  nit=0
  while np.any(np.abs(llhprev-llhcurr)>tol):
    
    t1 = time.time()
    # Change current likelihood to previous
    llhprev = llhcurr
    
    # Work out how many voxels are left
    nv_iter = XtY.shape[0]
    
    
    #---------------------------------------------------------------------------
    # Update beta
    beta = np.linalg.solve(XtX - XtZ @ DinvIplusZtZD @ ZtX, XtY - XtZ @ DinvIplusZtZD @ ZtY)
    
    # Update sigma^2
    ete = ssr3D(YtX, YtY, XtX, beta)
    Zte = ZtY - (ZtX @ beta)
    sigma2 = 1/n*(ete - Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte).reshape(nv_iter)
    
    # Update D_k
    counter = 0
    for k in np.arange(len(nparams)):
      
      # Work out update amount
      #update_k = forceSym3D(dupDuptMatdict[k] @ np.linalg.inv(get_mat_covdlDk(k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict))) @ dupInvDupMatdict[k] @ get_vec_2dlDk(k, nlevels, nparams, sigma2, ZtZ, Zte, DinvIplusZtZD)
      
      #print(forceSym3D(np.linalg.inv(get_covdldDk1Dk23D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict))).shape)
      #print(mat2vech3D(get_dldDk3D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD)).shape)
      #update = forceSym3D(np.linalg.inv(get_covdldDk1Dk23D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict))) @ mat2vech3D(get_dldDk3D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD))
      #update_k2 = mat2vec3D(vech2mat3D(update_k2))
      #print(update.shape)
      
      #print('1==2',update_k[1,:,:]-update_k2[1,:,:])
      
      
      update_p = forceSym3D(np.linalg.inv(get_mat_covdlDk(k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict))) @ get_vec_2dlDk(k, nlevels, nparams, sigma2, ZtZ, Zte, DinvIplusZtZD)
      #print(update_p.shape)

      #print('3==2',update_k3[1,:,:]-update_k2[1,:,:])
      
      # Multiply by stepsize
      update_p = np.einsum('i,ijk->ijk',lam, update_p)
      #update = np.einsum('i,ijk->ijk',lam, update)
      
      # Update D_k
      #Ddict[k] = makeDnnd3D(vec2mat3D(mat2vec3D(Ddict[k]) + update_p))
      Ddict[k] = makeDnnd3D(vech2mat3D(mat2vech3D(Ddict[k]) + update_p))
      
      # Add D_k back into D and recompute DinvIplusZtZD
      for j in np.arange(nlevels[k]):

        D[:, Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
        counter = counter + 1
      
      #D = getDfromDict3D(Ddict,nparams,nlevels)
      
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
    savedparams[indices_ConDuringIt,0:p,:]=beta[localconverged,:,:]
    savedparams[indices_ConDuringIt,p:(p+1),:]=sigma2[localconverged].reshape(sigma2[localconverged].shape[0],1,1)
    
    for k in np.arange(len(nparams)):
      
      # Get vech form of D_k
      vech_Dk = mat2vech3D(Ddict[k][localconverged,:,:])
      
      # Make sure it has correct shape (i.e. shape (num voxels converged, num params for factor k squared, 1))
      vech_Dk = vech_Dk.reshape(len(localconverged),nparams[k]*(nparams[k]+1)//2,1)
      savedparams[indices_ConDuringIt,FishIndsDk[k]:FishIndsDk[k+1],:]=vech_Dk
      
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



