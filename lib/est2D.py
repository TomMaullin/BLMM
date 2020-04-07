import time
import os
import numpy as np
import scipy
from lib.npMatrix3d import *
from lib.npMatrix2d import *

def cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None):
  
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

  # Duplication matrices
  # ------------------------------------------------------------------------------
  invDupMatdict = dict()
  invElimMatdict = dict()
  elimMatdict = dict()
  comMatdict = dict()
  for i in np.arange(len(nparams)):

    invDupMatdict[i] = invDupMat2D(nparams[i])
    invElimMatdict[i] = scipy.sparse.lil_matrix(np.linalg.pinv(elimMat2D(nparams[i]).toarray()))
    comMatdict[i] = comMat2D(nparams[i],nparams[i])
    elimMatdict[i] = elimMat2D(nparams[i])
    
  # Initial estimates
  # ------------------------------------------------------------------------------

  if init_paramVector is not None:

    #print(init_paramVector.shape)
    beta = init_paramVector[0:p]
    sigma2 = init_paramVector[p:(p+1)][0,0]

    Ddict = dict()
    cholDict = dict()
    for k in np.arange(len(nparams)):

      cholDict[k] = vechTri2mat2D(init_paramVector[FishIndsDk[k]:FishIndsDk[k+1]])
      Ddict[k] = cholDict[k] @ cholDict[k].transpose()
    
    # Matrix version
    D = scipy.sparse.lil_matrix((q,q))
    counter = 0
    for k in np.arange(len(nparams)):
      for j in np.arange(nlevels[i]):

        D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
        counter = counter + 1

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
    cholDict = dict()

    for k in np.arange(len(nparams)):

      cholDict[k] = np.eye(nparams[k])
      Ddict[k] = np.eye(nparams[k])

    # Matrix version
    D = scipy.sparse.lil_matrix((q,q))
    t1 = time.time()
    counter = 0
    for k in np.arange(len(nparams)):
      for j in np.arange(nlevels[k]):

        D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
        counter = counter + 1

    t2 = time.time()
    print('toDict time: ', t2-t1)

  Zte = ZtY - (ZtX @ beta)

  # Inverse of (I+Z'ZD) multiplied by DIplusDZtZ 
  IplusZtZD = np.eye(q) + ZtZ @ D
  t1 = time.time()
  DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))
  t2 = time.time()
  print('inv time: ', t2-t1)

  # Step size lambda
  lam = 1
  
  # Initial log likelihoods
  llhprev = np.inf
  llhcurr = -np.inf
  
  # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
  ZtZmatdict = dict()
  for k in np.arange(len(nparams)):
    ZtZmatdict[k] = None

  # This will hold the permutations needed for the covariance between the
  # derivatives with respect to k
  permdict = dict()
  for k in np.arange(len(nparams)):
    permdict[str(k)] = None

  nit = 0
  while np.abs(llhprev-llhcurr)>tol:
    
    # Change current likelihood to previous
    llhprev = llhcurr

    # Update number of iterations
    nit = nit+1

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

      # Work out derivative
      if ZtZmatdict[k] is None:
        dldD,ZtZmatdict[k] = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
      else:
        dldD,_ = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

      # Work out update amount
      if permdict[str(k)] is None:
        covdldDk,permdict[str(k)] = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, perm=None)
      else:
        covdldDk,_ = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, perm=permdict[str(k)])

      # We need to modify by multiplying by this matrix
      chol_mod = elimMatdict[k] @ (scipy.sparse.identity(nparams[k]**2) + comMatdict[k]) @ scipy.sparse.kron(cholDict[k],np.eye(nparams[k])) @ elimMatdict[k].transpose()
      
      # Transform to cholesky
      dldcholk = chol_mod.transpose() @ mat2vech2D(dldD)
      covdldcholk = chol_mod.transpose() @ covdldDk @ chol_mod

      update = lam*forceSym2D(np.linalg.inv(covdldcholk)) @ dldcholk
    
      # Update D_k and chol
      cholDict[k] = vechTri2mat2D(mat2vechTri2D(cholDict[k]) + update)

      Ddict[k] = cholDict[k] @ cholDict[k].transpose()

      # Add D_k back into D and recompute DinvIplusZtZD
      for j in np.arange(nlevels[k]):

        D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
        counter = counter + 1
      
      # Inverse of (I+Z'ZD) multiplied by D
      IplusZtZD = np.eye(q) + (ZtZ @ D)
      DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))

    # Update the step size
    llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
    if llhprev>llhcurr:
      lam = lam/2

  paramVector = np.concatenate((beta, sigma2))
  for k in np.arange(len(nparams)):
    paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

  bvals = DinvIplusZtZD @ Zte

  return(paramVector, bvals, nit)


def FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None):
  
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

      Ddict[k] = makeDnnd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nparams, invDupMatdict))
      
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
  
  # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
  ZtZmatdict = dict()
  for k in np.arange(len(nparams)):
    ZtZmatdict[k] = None

  # This will hold the permutations needed for the covariance between the
  # derivatives with respect to k1 and k2
  permdict = dict()
  for k1 in np.arange(len(nparams)):
    for k2 in np.arange(len(nparams)):
      permdict[str(k1)+str(k2)] = None

  nit = 0
  while np.abs(llhprev-llhcurr)>tol:
    
    #print('nit', counter)
    nit = nit+1
    
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
      # Store it in the dictionary# Store it in the dictionary
      if ZtZmatdict[k] is None:
        dldDdict[k],ZtZmatdict[k] = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
      else:
        dldDdict[k],_ = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])


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
      if ZtZmatdict[k] is None:
        covdldDksigma2,ZtZmatdict[k] = get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, ZtZmat=None)
      else:
        covdldDksigma2,_ = get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, ZtZmat=ZtZmatdict[k])

      # Assign to the relevant block
      FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldDksigma2.reshape(FishIndsDk[k+1]-FishIndsDk[k])
      FisherInfoMat[FishIndsDk[k]:FishIndsDk[k+1],p] = FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]].transpose()

    # Add dl/dD covariance
    for k1 in np.arange(len(nparams)):

      for k2 in np.arange(k1+1):

        IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
        IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

        # Get covariance between D_k1 and D_k2 
        if permdict[str(k1)+str(k2)] is None:
          FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],permdict[str(k1)+str(k2)] = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict,perm=None)
        else:
          FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],_ = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict,perm=permdict[str(k1)+str(k2)])

        # Get covariance between D_k1 and D_k2 
        FisherInfoMat[np.ix_(IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(IndsDk1, IndsDk2)].transpose()

    paramVector = np.concatenate((beta, np.array([[sigma2]])))
    derivVector = np.concatenate((dldB, dldsigma2))

    for k in np.arange(len(nparams)):

      paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))
      derivVector = np.concatenate((derivVector, mat2vech2D(dldDdict[k])))

    FisherInfoMat = forceSym2D(FisherInfoMat)

    paramVector = paramVector + lam*(np.linalg.inv(FisherInfoMat) @ derivVector)

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
  
  return(paramVector, bvals, nit)


def pFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None):
  
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
  tnp = np.int32(p + 1 + np.sum(nparams**2))

  # Indices for submatrics corresponding to Dks
  FishIndsDk = np.int32(np.cumsum(nparams**2) + p + 1)
  FishIndsDk = np.insert(FishIndsDk,0,p+1)

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

      Ddict[k] = makeDnnd2D(vec2mat2D(init_paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
      
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

      Ddict[k] = makeDnnd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nparams, invDupMatdict))
      
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
  
  # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
  ZtZmatdict = dict()
  for k in np.arange(len(nparams)):
    ZtZmatdict[k] = None

  # This will hold the permutations needed for the covariance between the
  # derivatives with respect to k1 and k2
  permdict = dict()
  for k1 in np.arange(len(nparams)):
    for k2 in np.arange(len(nparams)):
      permdict[str(k1)+str(k2)] = None

  nit = 0
  while np.abs(llhprev-llhcurr)>tol:
    

    nit = nit+1
    
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
      if ZtZmatdict[k] is None:
        dldDdict[k],ZtZmatdict[k] = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
      else:
        dldDdict[k],_ = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])


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
      if ZtZmatdict[k] is None:
        covdldDksigma2,ZtZmatdict[k] = get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True, ZtZmat=None)
      else:
        covdldDksigma2,_ = get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True, ZtZmat=ZtZmatdict[k])

      # Assign to the relevant block
      FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]] = covdldDksigma2.reshape(FishIndsDk[k+1]-FishIndsDk[k])
      FisherInfoMat[FishIndsDk[k]:FishIndsDk[k+1],p] = FisherInfoMat[p, FishIndsDk[k]:FishIndsDk[k+1]].transpose()

    # Add dl/dD covariance
    for k1 in np.arange(len(nparams)):

      for k2 in np.arange(k1+1):

        IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
        IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

        # Get covariance between D_k1 and D_k2 
        if permdict[str(k1)+str(k2)] is None:
          FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],permdict[str(k1)+str(k2)] = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict,vec=True,perm=None)
        else:
          FisherInfoMat[np.ix_(IndsDk1, IndsDk2)],_ = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict,vec=True,perm=permdict[str(k1)+str(k2)])

        FisherInfoMat[np.ix_(IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(IndsDk1, IndsDk2)].transpose()

    paramVector = np.concatenate((beta, np.array([[sigma2]])))
    derivVector = np.concatenate((dldB, dldsigma2))

    for k in np.arange(len(nparams)):

      paramVector = np.concatenate((paramVector, mat2vec2D(Ddict[k])))
      derivVector = np.concatenate((derivVector, mat2vec2D(dldDdict[k])))

    FisherInfoMat = forceSym2D(FisherInfoMat)

    paramVector = paramVector + lam*(np.linalg.inv(FisherInfoMat) @ derivVector)
    
    if sigma2<0:

      sigspos[z]=1
      sigma2 = np.maximum(sigma2,1e-6)

    #print(paramVector)
    beta = paramVector[0:p]
    sigma2 = paramVector[p:(p+1)][0,0]

    for k in np.arange(len(nparams)):

      Ddict[k] = makeDnnd2D(vec2mat2D(paramVector[FishIndsDk[k]:FishIndsDk[k+1]]))
      
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

  # Convert to vech representation.
  # Indices for submatrics corresponding to Dks
  IndsDk = np.int32(np.cumsum(nparams*(nparams+1)//2) + p + 1)
  IndsDk = np.insert(IndsDk,0,p+1)

  paramVector_reshaped = np.zeros((np.sum(nparams*(nparams+1)//2) + p + 1,1))
  paramVector_reshaped[0:(p+1)]=paramVector[0:(p+1)].reshape(p+1,1)
  for k in np.arange(len(nlevels)):

    paramVector_reshaped[IndsDk[k]:IndsDk[k+1]] = vec2vech2D(paramVector[FishIndsDk[k]:FishIndsDk[k+1]]).reshape(paramVector_reshaped[IndsDk[k]:IndsDk[k+1]].shape)

  
  return(paramVector_reshaped, bvals, nit)


def pSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None):
  
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
      
    # Matrix version
    D = scipy.sparse.lil_matrix((q,q))
    counter = 0
    for k in np.arange(len(nparams)):
      for j in np.arange(nlevels[k]):

        D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
        counter = counter + 1

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

      Ddict[k] = makeDnnd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nparams, invDupMatdict))
      
    # Matrix version
    D = scipy.sparse.lil_matrix((q,q))
    counter = 0
    for k in np.arange(len(nparams)):
      for j in np.arange(nlevels[k]):

        D[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
        counter = counter + 1

  Zte = ZtY - (ZtX @ beta)

  # Inverse of (I+Z'ZD) multiplied by DIplusDZtZ 
  IplusZtZD = np.eye(q) + ZtZ @ D
  DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))

  # Step size lambda
  lam = 1
  
  # Initial log likelihoods
  llhprev = np.inf
  llhcurr = -np.inf
  
  # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
  ZtZmatdict = dict()
  for k in np.arange(len(nparams)):
    ZtZmatdict[k] = None

  # This will hold the permutations needed for the covariance between the
  # derivatives with respect to k
  permdict = dict()
  for k in np.arange(len(nparams)):
    permdict[str(k)] = None

  nit = 0
  while np.abs(llhprev-llhcurr)>tol:
    
    # Change current likelihood to previous
    llhprev = llhcurr

    #print(nit)
    nit = nit+1

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

      # Work out derivative
      if ZtZmatdict[k] is None:
        dldD,ZtZmatdict[k] = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
      else:
        dldD,_ = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

      
      # Work out update amount
      if permdict[str(k)] is None:
        covdldDk,permdict[str(k)] = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True, perm=None)
      else:
        covdldDk,_ = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=True, perm=permdict[str(k)])

      # Work out update amount
      update = lam*forceSym2D(np.linalg.inv(covdldDk)) @ mat2vec2D(dldD)
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
      DinvIplusZtZD = forceSym2D(D @ scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(IplusZtZD)))

    # Update the step size
    llhcurr = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)[0,0]
    if llhprev>llhcurr:
      lam = lam/2

  paramVector = np.concatenate((beta, sigma2))
  for k in np.arange(len(nparams)):
    paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

  bvals = DinvIplusZtZD @ Zte
    
  return(paramVector, bvals, nit)


def SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, tol, n, init_paramVector=None):
  
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

      Ddict[k] = makeDnnd2D(initDk2D(k, ZtZ, Zte, sigma2, nlevels, nparams, invDupMatdict))
      
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
  
  # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
  ZtZmatdict = dict()
  for k in np.arange(len(nparams)):
    ZtZmatdict[k] = None

  # This will hold the permutations needed for the covariance between the
  # derivatives with respect to k
  permdict = dict()
  for k in np.arange(len(nparams)):
    permdict[str(k)] = None

  nit = 0

  while np.abs(llhprev-llhcurr)>tol:

    nit = nit + 1

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
      
      # Work out derivative
      if ZtZmatdict[k] is None:
        dldD,ZtZmatdict[k] = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
      else:
        dldD,_ = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

      # Work out update amount
      if permdict[str(k)] is None:
        covdldDk,permdict[str(k)] = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict,perm=None)
      else:
        covdldDk,_ = get_covdldDk1Dk22D(k, k, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, perm=permdict[str(k)])

      update = lam*forceSym2D(np.linalg.inv(covdldDk)) @ mat2vech2D(dldD)
      
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
  
  return(paramVector, bvals, nit)
