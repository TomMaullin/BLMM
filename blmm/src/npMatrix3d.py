import numpy as np
from scipy import stats
from blmm.src.npMatrix2d import faclev_indices2D, fac_indices2D, permOfIkKkI2D, dupMat2D

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Developers Notes:
#
# - Tom Maullin (12/11/2019)
#   Apologies for the poorly typeset equations. Once I got into this project 
#   it really helped me having some form of the equations handy but I 
#   understand this may not be as readable for developers new to the code. For
#   nice latexed versions of the documentation here please see the google 
#   colab notebooks here:
#     - PeLS: https://colab.research.google.com/drive/1add6pX26d32WxfMUTXNz4wixYR1nOGi0
#     - FS: https://colab.research.google.com/drive/12CzYZjpuLbENSFgRxLi9WZfF5oSwiy-e
#     - GS: https://colab.research.google.com/drive/1sjfyDF_EhSZY60ziXoKGh4lfb737LFPD
# - Tom Maullin (12/11/2019)
#   This file contains "3D" versions of all functions given in `2dtools.py`. 
#   By 3D, I mean, where `2dtools.py` would take a matrix as input and do 
#   something to it, `3dtools.py` will take as input a 3D array (stack of 
#   matrices) and perform the operation to all matrices in the last 2 
#   dimensions. (Note: A lot of the documentation below is identical to 
#   2dtools so beware this distinction!).
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ============================================================================
#
# This function performs a broadcasted kronecker product.
# 
# The code was adapted from:
#   https://stackoverflow.com/questions/57259557/kronecker-product-of-matrix-array
#
# ============================================================================
def kron3D(A,B):

  i1,j,k = A.shape
  i2,l,m = B.shape

  if i1==i2:
    kronRes = np.einsum("ijk,ilm->ijlkm",A,B).reshape(i1,j*l,k*m)
  elif i1==1 or i2==1:
    kronRes = np.einsum("ijk,nlm->injlkm",A,B).reshape(i1*i2,j*l,k*m)
  else:
    raise ValueError('Incompatible dimensions in kron3D.')

  # Return
  return(kronRes)


# ============================================================================
#
# This function performs a broadcasted kronecker product.
# 
# The code was adapted from:
#   https://stackoverflow.com/questions/57259557/kronecker-product-of-matrix-array
#
# ============================================================================
def kron4D(A,B):

  i1,i3,j,k = A.shape
  i2,i4,l,m = B.shape

  if i1==i2 and i3==i4:
    kronRes = np.einsum("bijk,bilm->bijlkm",A,B).reshape(i1,i3,j*l,k*m)
  elif i1==1 or i2==1:
    kronRes = np.einsum("ijk,nlm->injlkm",A,B).reshape(i1*i2,i3,j*l,k*m)
  else:
    raise ValueError('Incompatible dimensions in kron3D.')

  # Return
  return(kronRes)

# ============================================================================
#
# This function takes in a matrix and vectorizes it (i.e. transforms it
# to a vector of each of the columns of the matrix stacked on top of
# one another).
#
# ============================================================================
def mat2vec3D(matrix):
  
  #Return vectorised matrix
  vec=matrix.transpose(0,2,1).reshape(matrix.shape[0],matrix.shape[1]*matrix.shape[2],1)

  # Return
  return(vec)


# ============================================================================
#
# This function takes in a (symmetric, square) matrix and half-vectorizes
# it (i.e. transforms it to a vector of each of the columns of the matrix,
# below and including the diagonal, stacked on top of one another).
#
# ============================================================================
def mat2vech3D(matrix):

  # Number of voxels, v
  v = matrix.shape[0]
  
  # Get lower triangular indices
  rowinds, colinds = np.tril_indices(matrix.shape[1]) 
  
  # Number of unique elements, nc
  nc = len(rowinds)
  
  # They're in the wrong order so we need to order them
  # To do this we first hash them
  indhash = colinds*matrix.shape[1]+rowinds
  
  # Sort permutation
  perm=np.argsort(indhash)
  
  # Return vectorised half-matrix
  vech=matrix[:,rowinds[perm],colinds[perm]].reshape((v,nc,1))
  
  # Return
  return(vech)

# ============================================================================
#
# This function takes in a stack of vectors and returns the corresponding
# matrix forms treating the elements of the vectors as the elements of lower
# halves of those matrices.
#
# ============================================================================
def vech2mat3D(vech):
  
  # Number of voxels
  v = vech.shape[0]
  
  # dimension of matrix
  n = np.int64((-1+np.sqrt(1+8*vech.shape[1]))/2)
  matrix = np.zeros((v,n,n))
  
  # Get lower triangular indices
  rowinds, colinds = np.tril_indices(n)
  
  # They're in the wrong order so we need to order them
  # To do this we first hash them
  indhash = colinds*n+rowinds
  
  # Sort permutation
  perm=np.argsort(indhash)
  
  # Assign values to lower half
  matrix[:,rowinds[perm],colinds[perm]] = vech.reshape(vech.shape[0],vech.shape[1])
  
  # Assign values to upper half
  matrix[:,colinds[perm],rowinds[perm]] = vech.reshape(vech.shape[0],vech.shape[1])
  
  # Return vectorised half-matrix
  return(matrix)


# ============================================================================
#
# This function maps the vector created by stacking the columns of a matrix on
# top of one another to it's corresponding square matrix.
#
# ============================================================================
def vec2mat3D(vec):
  
  # Return matrix
  matrix=vec.reshape(vec.shape[0], np.int64(np.sqrt(vec.shape[1])),np.int64(np.sqrt(vec.shape[1]))).transpose(0,2,1)

  return(matrix)

# ============================================================================
#
# This function takes in a matrix X and returns (X+X')/2 (forces it to be
# symmetric).
#
# ============================================================================
# Developer NOTE: DO NOT USE NUMBA ON THIS FUNCTION - NUMBA MESSES IT UP FOR 
# UNKNOWN REASONS
def forceSym3D(x):

  # Force it to be symmetric
  matrix=(x+x.transpose((0,2,1)))/2

  return(matrix)

# ============================================================================
#
# The function below calculates the sum of the square residuals, e'e,
# using the below formula:
# 
# e'e = (Y-X\beta)'(Y-X\beta) 
#     = Y'Y - 2Y'X\beta + \beta'X'X\beta
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
def ssr3D(YtX, YtY, XtX, beta):
  
  # Return the sum of squared residuals
  ssr=YtY - 2*YtX @ beta + beta.transpose((0,2,1)) @ XtX @ beta

  return(ssr)


# ============================================================================
#
# This function takes in a dictionary, `Ddict`, in which entry `k` is a stack 
# of the kth diagonal block for every voxel.
#
# ============================================================================
def getDfromDict3D(Ddict, nraneffs, nlevels):
  
  # Get number of voxels
  v = Ddict[0].shape[0]
  
  # Work out indices (there is one block of D per level)
  inds = np.zeros(np.sum(nlevels)+1)
  counter = 0
  for k in np.arange(len(nraneffs)):
    for j in np.arange(nlevels[k]):
      inds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
      counter = counter + 1
      
  
  # Last index will be missing so add it
  inds[len(inds)-1]=inds[len(inds)-2]+nraneffs[-1]
  
  # Make sure indices are ints
  inds = np.int64(inds)
  
  # Initial D
  D = np.zeros((v,np.sum(nraneffs*nlevels),np.sum(nraneffs*nlevels)))

  counter = 0
  for k in np.arange(len(nraneffs)):
    for j in np.arange(nlevels[k]):

      D[:, inds[counter]:inds[counter+1], inds[counter]:inds[counter+1]] = Ddict[k]
      counter = counter + 1

  return(D)


# ============================================================================
#
# The below function returns the OLS estimator for \beta, given by:
#
# \bethat=(X'X)^(-1)X'Y
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `XtX`: The design matrix transposed and multiplied by itself (X'X in the
#          above notation)
# - `XtY`: The design matrix transposed and multiplied by the response vector
#          (X'Y in the above notation).
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `beta`: The OLS estimate of \beta (\betahat in the above notation).
#
# ============================================================================
def initBeta3D(XtX, XtY):
  
  # Get the beta estimator
  beta = np.linalg.solve(XtX,XtY)

  # Return the result
  return(beta)


# ============================================================================
#
# The function below returns an initial estimate for the Fixed Effects
# Variance, \sigma^2. The estimator used is based on the suggested OLS
# estimator in Demidenko (2012) and is given by:
#
# \sigmahat^2=1/n(Y-X\betahat)'(Y-X\betahat)
#            =1/n e'e
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `ete`: The sum of square residuals (e'e in the above notation).
# - `n`: The total number of observations (potentially spatially varying).
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `sigma2`: The OLS estimate of \sigma^2 (\sigmahat^2 in the above notation).
#
# ============================================================================
def initSigma23D(ete, n):

  if hasattr(n, "ndim"):

    if np.prod(n.shape) > 1:

      n = n.reshape(ete[:,0,0].shape)

  sigma2=1/n*ete[:,0,0]

  # Return the OLS estimate of sigma
  return(sigma2)


# ============================================================================
#
# The function below returns an initial estimate for the Random Effects 
# Variance matrix for the $k^{th}$ grouping factor, $D_k$. The estimator used
# is an adaption of the suggested estimator in Demidenko (2012) and is given by:
#
# vec(Dhat_k)=[sum_(j=1)^(l_k)(Z_(k,j)'Z_(k,j)) kron (Z_(k,j)'Z_(k,j))]^(-1)*
#              vec(\sum_(j=1)^(l_k)[\sigma^(-2)Z_(k,j)'ee'Z_(k,j)-Z_(k,j)'Z_(k,j)])
#
# Or:
# 
# Dhat_k=matrix([sum_(j=1)^(l_k)(Z_(k,j)'Z_(k,j)) kron (Z_(k,j)'Z_(k,j))]^(-1)*
#        vec(sum_(j=1)^(l_k)[\sigma^(-2)Z_(k,j)'ee'Z_(k,j) - Z_(k,j)'Z_(k,j)]))
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `k`: The grouping factor we wish to estimate D for (k in the above
#        notation)
# - `lk`: The number of levels belonging to grouping factor k ($l_k$ in the
#         above notation).
# - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). If we are looking at a one random factor one
#          random effect design the variable ZtZ only holds the diagonal 
#          elements of the matrix Z'Z.
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The OLS estimate of \sigma^2 (\sigma^2$ in the above notation).
# - `dupMatTdict`: A dictionary of transpose duplication matrices such that 
#                   `dupMatTdict[k]` = DupMat_k'.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `Dkest`: The inital estimate of D_k (Dhat_k in the above notation).
#
# ============================================================================
def initDk3D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict):
  
  # Number of voxels v
  v = Zte.shape[0]

  # Number of random factors
  r = len(nraneffs)

  # Small check on sigma2
  if len(sigma2.shape) > 1:

    sigma2 = sigma2.reshape(sigma2.shape[0])

  # If we have only one factor and one random effect computation can be
  # sped up a lot
  if r == 1 and nraneffs[0]==1:

    # Work out block size (should be 1)
    qk = nraneffs[k]
    pttn = np.array([qk,1])

    # Work out Z'ee'Z/sigma^2 - Z'Z
    invSig2ZteetZminusZtZ = np.einsum('i,ijk->ijk',1/sigma2,sumAijBijt3D(Zte, Zte, pttn, pttn)) - np.sum(ZtZ,axis=1).reshape(ZtZ.shape[0],1,1)

  else:

    # Initalize D to zeros
    invSig2ZteetZminusZtZ = np.zeros((Zte.shape[0],nraneffs[k],nraneffs[k]))

    # First we work out the derivative we require.
    for j in np.arange(nlevels[k]):
      
      # Indices for factor k level j
      Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

      # This can also be performed faster in the one factor, multiple random effect
      # case by using only the diagonal blocks of DinvIplusZtZD 
      if r == 1 and nraneffs[0] > 1:

        # Work out Z_(k, j)'Z_(k, j)
        ZkjtZkj = ZtZ[:,:,Ikj]

      else:

        # Work out Z_(k, j)'Z_(k, j)
        ZkjtZkj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)]

      # Work out Z_(k,j)'e
      Zkjte = Zte[:, Ikj,:]

      if j==0:
        
        # Add first \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
        invSig2ZteetZminusZtZ = np.einsum('i,ijk->ijk',1/sigma2,(Zkjte @ Zkjte.transpose(0,2,1))) - ZkjtZkj
        
      else:
        
        # Add next \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
        invSig2ZteetZminusZtZ = invSig2ZteetZminusZtZ + np.einsum('i,ijk->ijk',1/sigma2,(Zkjte @ Zkjte.transpose(0,2,1))) - ZkjtZkj

  # Again, if we have only one factor and one random effect computation can 
  # be sped up a lot
  if r == 1 and nraneffs[0]==1:

    # Information matrix for initial estimate
    infoMat = np.sum(ZtZ**2,axis=1).reshape((ZtZ.shape[0],1,1))

  # This can also be performed faster in the one factor, multiple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD 
  elif r == 1 and nraneffs[0] > 1:

    # Sum of Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j), as for the one factor model
    # off diagonal blocks cancel to zero
    for j in np.arange(nlevels[k]):

      Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

      # Work out Z_(k, j)'Z_(k, j)
      ZkjtZkj = ZtZ[:,:,Ikj]

      if j==0:
        
        # Add first Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
        ZtZkronZtZ = kron3D(ZkjtZkj,ZkjtZkj.transpose(0,2,1))
     
      else:
        
        # Add next Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
        ZtZkronZtZ = ZtZkronZtZ + kron3D(ZkjtZkj,ZkjtZkj.transpose(0,2,1))

    # Work out information matrix as:
    # Dup_k @ (sum_j Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)) @ Dup_k'
    infoMat = dupMatTdict[k] @ ZtZkronZtZ @ dupMatTdict[k].transpose()

  else:

    # Double sum of Z_(k,i)'Z_(k,j) kron Z_(k,i)'Z_(k,j)
    for j in np.arange(nlevels[k]):

      for i in np.arange(nlevels[k]):
        
        Iki = faclev_indices2D(k, i, nlevels, nraneffs)
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

        # Work out Z_(k, j)'Z_(k, j)
        ZkitZkj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Iki,Ikj)]
        
        if j==0 and i==0:
          
          # Add first Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
          ZtZkronZtZ = kron3D(ZkitZkj,ZkitZkj.transpose(0,2,1))
       
        else:
          
          # Add next Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
          ZtZkronZtZ = ZtZkronZtZ + kron3D(ZkitZkj,ZkitZkj.transpose(0,2,1))

    # Work out information matrix as:
    # Dup_k @ (sum_i sum_j Z_(k,i)'Z_(k,j) kron Z_(k,i)'Z_(k,j)) @ Dup_k'
    infoMat = dupMatTdict[k] @ ZtZkronZtZ @ dupMatTdict[k].transpose()

  # Work out the final term.
  Dkest = vech2mat3D(np.linalg.solve(infoMat, dupMatTdict[k] @ mat2vec3D(invSig2ZteetZminusZtZ)))
  
  return(Dkest)

# ============================================================================
#
# The below function takes in a covariance matrix D and finds nearest
# projection onto the space of non-negative definite matrices D_+. It uses the
# following method taken from Demidenko (2012), page 105:
#
# If D is non-negative definite and has eigenvalue decomposition
# D=P\Lambda P' it's closest projection into D_+ is defined by the matrix
# below:
#
# Dhat_+ = P\Lambda_+P'
#
# Where \Lambda_+ is defined by the elementwise maximum of \Lambda and 0; i.e.
# \Lambda_+(i,j) = max(\Lambda_+(i,j),0).
#
# Note: This is not to be confused with the generalized inverse of the
# duplication matrix, also denoted with a D+.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `D`: A square symmetric matrix. (Note: This only takes in D as a square
#        matrix. Unlike other functions, if D is just a vector of the diagonal
#        elements of a square matrix, this code will not know what to do).
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `D_nnd`: The nearest projection of D onto the space of non-negative
#            definite matrices D_+.
#
# ============================================================================
def makeDnnd3D(D):
  
  # Check if we have negative eigenvalues
  if not np.all(np.linalg.eigvals(D)>0):
  
    # If we have negative eigenvalues
    eigvals,eigvecs = np.linalg.eigh(D)
    
    # Work out elementwise max of lambda and 0
    lamplus = np.zeros((eigvals.shape[0], eigvals.shape[1], eigvals.shape[1]))
    diag = np.arange(eigvals.shape[1])
    lamplus[:, diag, diag] = np.maximum(eigvals,0)
    
    # Work out D+
    D_nnd = eigvecs @ lamplus @ np.linalg.inv(eigvecs)
    
  else:
    
    # D is already non-negative in this case
    D_nnd = D
    
  return(D_nnd)


# ============================================================================
# This function returns the log likelihood of (\beta, \sigma^2, D) which is
# given by the below equation:
#
# l(\beta,\sigma^2,D) = -0.5(nln(\sigma^2) + ln|I+Z'ZD| +
#                       \sigma^(-2)(e'e-e'ZD(I+Z'ZD)^(-1)Z'e))
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `n`: The total number of observations (potentially spatially varying).
# - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). If we are looking at a one random factor one
#          random effect design the variable ZtZ only holds the diagonal 
#          elements of the matrix Z'Z.
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `ete`: The OLS residuals transposed and then multiplied by themselves
#          (e'e=(Y-X\beta)'(Y-X\beta) in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `D`: The random effects variance-covariance matrix (D in the above
#        notation). Note in the one random factor, one random effect use
#        case D can be set to none as the Ddict representation is used 
#        instead.
# - `Ddict`: Dictionary version of the random effects variance-covariance
#            matrix.
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `llh`: The log likelihood of (\beta, \sigma^2, D) (l(\beta,\sigma^2,D) in
#          the above notation).
#
# ============================================================================
def llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D, Ddict, nlevels, nraneffs, reml=False, XtX=0, XtiVX=0):

  # Number of random effects and number of voxels
  r = len(nlevels)
  v = ete.shape[0]

  # Number of fixed effects
  if reml:
    p = XtX.shape[-1]

  # Reshape n if neccesary
  if hasattr(n, "ndim"):

    if np.prod(n.shape) > 1:

      n = n.reshape(sigma2.shape)

  # If we have only one factor and one random effect computation can be
  # sped up a lot
  if r == 1 and nraneffs[0]==1:
    
    # Work out the diagonal entries of I+Z'ZD (we assume ZtZ is already just the
    # diagonal elements in this use case)
    DiagIplusZtZD = 1 + ZtZ*Ddict[0].reshape(v,1)

    # Get the log of them
    logDiagIplusZtZD = np.log(DiagIplusZtZD)

    # The result should be the log of their sum.
    logdet = np.sum(logDiagIplusZtZD,axis=1).reshape(ete.shape[0])

  # In the one factor multiple random effect setting this computation is 
  # also quicker as DinvIplusZtZD is block diagonal.
  elif r == 1 and nraneffs[0]>1:

    # q0, l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

    # Reshape D_0
    IplusDZtZ = Ddict[0].reshape(v,1,q0,q0)

    # Multiply by ZtZ
    IplusDZtZ = IplusDZtZ @ ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0],l0,q0,q0)

    # Add identitiy
    IplusDZtZ = np.eye(q0) + IplusDZtZ

    # Take log determinant
    logdet = np.linalg.slogdet(IplusDZtZ)

    # Sum across levels
    logdet = np.sum(logdet[0]*logdet[1], axis=1).reshape(ete.shape[0])

  # Else we have to use niave computation
  else:

    # Work out log|I+ZtZD|
    logdet = np.linalg.slogdet(np.eye(ZtZ.shape[1]) + D @ ZtZ)
    logdet = (logdet[0]*logdet[1]).reshape(ete.shape[0])

  # Work out -1/2(nln(sigma^2) + ln|I+DZ'Z|)
  if reml==False:
    firstterm = -0.5*(n*np.log(sigma2)).reshape(ete.shape[0]) - 0.5*logdet
  else:
    p = XtX.shape[1]
    firstterm = -0.5*((n-p)*np.log(sigma2)).reshape(ete.shape[0]) -0.5*logdet


  # If we have only one factor and one random effect computation can be
  # sped up a lot
  if r == 1 and nraneffs[0]==1:

    secondterm = -0.5*np.einsum('i,ijk->ijk',(1/sigma2).reshape(ete.shape[0]),(ete - forceSym3D(Zte.transpose((0,2,1)) @ np.einsum('ij,ijk->ijk',DinvIplusZtZD, Zte)))).reshape(ete.shape[0])

  # This can also be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0] > 1:

    # Reshape DinvIplusZtZD appropriately
    DinvIplusZtZDZte = DinvIplusZtZD.transpose(0,2,1).reshape(v,l0,q0,q0)

    # Multiply by Zte
    DinvIplusZtZDZte = DinvIplusZtZDZte @ Zte.reshape(v,l0,q0,1)    

    # Reshape appropriately
    DinvIplusZtZDZte = DinvIplusZtZDZte.reshape(v,q0*l0,1)

    # Calculate second term
    secondterm = -0.5*np.einsum('i,ijk->ijk',(1/sigma2).reshape(ete.shape[0]),(ete - forceSym3D(Zte.transpose((0,2,1)) @ DinvIplusZtZDZte))).reshape(ete.shape[0])

  else:

    # Work out sigma^(-2)*(e'e - e'ZD(I+Z'ZD)^(-1)Z'e)
    secondterm = -0.5*np.einsum('i,ijk->ijk',(1/sigma2).reshape(ete.shape[0]),(ete - forceSym3D(Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte))).reshape(ete.shape[0])

  # Work out the log likelihood
  llh = (firstterm + secondterm).reshape(ete.shape[0])

  if reml:

    # Get log of determinant
    logdet = np.linalg.slogdet(XtiVX)
    
    # Take from llh
    llh = llh - 0.5*logdet[0]*logdet[1]

  # Return result
  return(llh)



# ============================================================================
#
# The below function calculates the matrix D(I+Z'ZD)^(-1). Whilst in general,
# this computation can't be streamlined, many common usecases are easier to
# calculate. This function takes those into account.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `Ddict`: a dictionary in which entry `k` is a 3D array of the kth diagonal 
#            block of D for every voxel.
# - `D`: The matrix equivalent of Ddict. Note in the one random factor, one 
#        random effect use case D can be set to none as the Ddict
#        representation is used instead.
# - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). If we are looking at a one random factor one
#          random effect design the variable ZtZ only holds the diagonal 
#          elements of the matrix Z'Z.
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
#
# ============================================================================
def get_DinvIplusZtZD3D(Ddict, D, ZtZ, nlevels, nraneffs):

  # Work out how many factors we're looking at
  r = len(nlevels)
  q = ZtZ.shape[-1]
  v = Ddict[0].shape[0]

  # If one factor and one random effect, Z'Z is diagonal
  if r == 1 and nraneffs[0]==1:

    # In this case Z'Z and D are diagonal so we can
    # do 1/(I+Z'ZD)_ii to get the inverses.

    # Work out Diag(Z'ZD) (We assume ZtZ is already diagonal here)
    DiagZtZD = ZtZ*Ddict[0].reshape(v,1)

    # Work out Diag(D(I+Z'ZD)^(-1)) (In the 1 factor 1 raneff mode we only
    # need the diagonal elements)
    DinvIplusZtZD = Ddict[0].reshape(v,1)/(1+DiagZtZD)

  # If one factor and one random effect, Z'Z is block diagonal
  elif r == 1 and nraneffs[0]>1:

    # Get q0 and l0
    q0 = nraneffs[0]
    l0 = q//q0

    # Get I+Z'ZD
    IplusZtZD = np.eye(q0) + ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0], l0, q0, q0) @ Ddict[0].reshape(v,1,q0,q0)

    # Get D(I+Z'ZD)^(-1)
    DinvIplusZtZD = Ddict[0].reshape(v,1,q0,q0) @ np.linalg.pinv(IplusZtZD)

    # Force symmetry
    DinvIplusZtZD = 0.5*(DinvIplusZtZD+DinvIplusZtZD.transpose(0,1,3,2)) 

    # Reshape to flattened form
    DinvIplusZtZD = DinvIplusZtZD.reshape(v,l0*q0,q0).transpose(0,2,1)    

  else:

    DinvIplusZtZD = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

  return(DinvIplusZtZD)

# ============================================================================
# The below function calculates the derivative of the log likelihood with
# respect to \beta. This is given by the following equation:
#
# dl/(d\beta) = \sigma^(-2)X'(I+ZDZ')^(-1)(Y-X\beta)
#             = \sigma^(-2)X'(I-ZD(I+Z'ZD)^(-1)Z')(Y-X\beta)
#             = \sigma^(-2)X'(Y-X\beta)-X'ZD(I+Z'ZD)^(-1)Z'(Y-X\beta)
#             = \sigma^(-2)X'e-X'ZD(I+Z'ZD)^(-1)Z'e
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `XtZ`: The X matrix transposed and then multiplied by Z (X'Z in the above
#          notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `dldb`: The derivative of l with respect to \beta.
#
# ============================================================================
def get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte, nraneffs):

  # Number of random factors
  r = len(nraneffs)

  # In the one random factor one random effect setting this computation can be 
  # streamlined
  if r == 1 and nraneffs[0]==1:
    # Work out the derivative (Note: we leave everything as 3D for ease of future computation)
    deriv = np.einsum('i,ijk->ijk',1/sigma2, (Xte - (XtZ @ np.einsum('ij,ijk->ijk', DinvIplusZtZD, Zte))))

  # This can also be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0] > 1:

    # Number of voxels
    v = Zte.shape[0]

    # Number of random effects
    q = DinvIplusZtZD.shape[-1]

    # Get l0, q0
    q0 = nraneffs[0]
    l0 = q//q0

    # Reshape DinvIplusZtZD appropriately
    DinvIplusZtZDZte = DinvIplusZtZD.transpose(0,2,1).reshape(v,l0,q0,q0)

    # Multiply by Zte
    DinvIplusZtZDZte = DinvIplusZtZDZte @ Zte.reshape(v,l0,q0,1)    

    # Reshape appropriately
    DinvIplusZtZDZte = DinvIplusZtZDZte.reshape(v,q0*l0,1)

    # Work out the derivative (Note: we leave everything as 3D for ease of future computation)
    deriv = np.einsum('i,ijk->ijk',1/sigma2, (Xte - (XtZ @ DinvIplusZtZDZte)))

  else: 
    # Work out the derivative (Note: we leave everything as 3D for ease of future computation)
    deriv = np.einsum('i,ijk->ijk',1/sigma2, (Xte - (XtZ @ (DinvIplusZtZD @ Zte))))

  # Return the derivative
  return(deriv)

# ============================================================================
# The below function calculates the derivative of the log likelihood with
# respect to \sigma^2. This is given by the following equation:
#
# dl/(d\sigma^2) = -n/(2\sigma^2) + 1/(2\sigma^4)(Y-X\beta)'(I+ZDZ')^(-1)*
#                  (Y-X\beta)
#    = -n/(2\sigma^2) + 1/(2\sigma^4)e'(I+ZDZ')^(-1)e
#    = -n/(2\sigma^2) + 1/(2\sigma^4)e'(I-ZD(I+ZZ'D)^(-1)Z')e
#    = -n/(2\sigma^2) + 1/(2\sigma^4)(e'e-e'ZD(I+ZZ'D)^(-1)Z'e)
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `n`: The number of observations (potentially spatially varying).
# - `ete`: The OLS residuals transposed and then multiplied by themselves
#         (e'e=(Y-X\beta)'(Y-X\beta) in the above notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `dldsigma2`: The derivative of l with respect to \sigma^2.
#
# ============================================================================
def get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD, nraneffs, reml=False, p=0):
  
  # Make sure n is correct shape
  if hasattr(n, "ndim"):

    if np.prod(n.shape) > 1:

      n = n.reshape(sigma2.shape)

  # Number of random factors
  r = len(nraneffs)

  # In the one random factor one random effect setting this computation can be 
  # streamlined
  if r == 1 and nraneffs[0]==1:

    # Get e'(I+ZDZ')^(-1)e=e'e-e'ZD(I+Z'ZD)^(-1)Z'e
    etinvIplusZtDZe = ete - forceSym3D(Zte.transpose((0,2,1)) @ np.einsum('ij,ijk->ijk', DinvIplusZtZD, Zte))

  # This can also be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0] > 1:

    # Number of voxels v
    v = ete.shape[0]

    # Number of random effects
    q = DinvIplusZtZD.shape[-1]

    # lk and qk for the first factor (zero indexed)
    q0 = nraneffs[0]
    l0 = q//q0

    # Reshape DinvIplusZtZD appropriately
    DinvIplusZtZDZte = DinvIplusZtZD.transpose(0,2,1).reshape(v,l0,q0,q0)

    # Multiply by Zte
    DinvIplusZtZDZte = DinvIplusZtZDZte @ Zte.reshape(v,l0,q0,1)    

    # Reshape appropriately
    DinvIplusZtZDZte = DinvIplusZtZDZte.reshape(v,q0*l0,1)

    # Get e'(I+ZDZ')^(-1)e=e'e-e'ZD(I+Z'ZD)^(-1)Z'e
    etinvIplusZtDZe = ete - forceSym3D(Zte.transpose((0,2,1)) @ DinvIplusZtZDZte)

  else:

    # Get e'(I+ZDZ')^(-1)e=e'e-e'ZD(I+Z'ZD)^(-1)Z'e
    etinvIplusZtDZe = ete - forceSym3D(Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)

  # Get the derivative
  if not reml:
    deriv = -n/(2*sigma2) + np.einsum('i,ijk->ijk',1/(2*(sigma2**2)), etinvIplusZtDZe).reshape(sigma2.shape[0])
  else:
    deriv = -(n-p)/(2*sigma2) + np.einsum('i,ijk->ijk',1/(2*(sigma2**2)), etinvIplusZtDZe).reshape(sigma2.shape[0])

  return(deriv)


# ============================================================================
# The below function calculates the derivative of the log likelihood with
# respect to D_k, the random effects covariance matrix for factor k. This is
# given by the following equation:
#
# dl/(dD_k) = 0.5(sum_(j=1)^(l_k)(T_(k,j)u)(T_(k,j)u)'- ...
#             0.5sum_(j=1)^(l_k)T_(k,j)T_(k,j)'
#
# Where T_(i,j)=Z'_(i,j)(I+ZDZ')^(-0.5) and
# u=\sigma^{-1}(I+ZDZ')^(-0.5)(Y-X\beta)
# 
#    = 0.5\sigma^(-2)sum_(j=1)^(l_k)Z'_(k,j)(I+ZDZ')^(-1)ee'(I+ZDZ')^(-1)Z_(k,j)- ...
#      0.5sum_(j=1)^(l_k)Z'_(k,j)(I+ZDZ')^(-1)Z_(k,j)
#
#    = 0.5\sigma^(-2)sum_(j=1)^(l_k)(Z'_(k,j)e-...
#        Z'_(k,j)ZD(I+Z'ZD)^(-1)Z'e)(Z'_(k,j)e-...
#        Z'_(k,j)ZD(I+Z'ZD)^{-1}Z'e)' -...
#      0.5sum_(j=1)^(l_k)Z'_(k,j)Z_(k,j)-Z'_(k,j)ZD(I+Z'ZD)^(-1)Z'Z_(k,j)
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `k`: The factor we wish to estimate the derivative of the covariance
#        matrix of.
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). If we are looking at a one random factor one
#          random effect design the variable ZtZ only holds the diagonal 
#          elements of the matrix Z'Z.
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `ZtZmat`: The sum over j of Z_{(k,j)}'Z_{(k,j)}. This only need be 
#             calculated once so can be stored and re-entered for each
#             iteration.
# - `REML`: Backdoor option for restricted likelihood maximisation.
#           Currrently not maintained.
# - `ZtX`: The Z matrix transposed and then multiplied by X (Z'X in the
#          above notation). Only needed for REML.
# - `iXtiVX`: The inverse of X transposed and then multiplied by V inverse and 
#             then by X ((X'V^{-1}X)^{-1} in the above notation). Only needed
#             for REML.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `dldDk`: The derivative of l with respect to D_k.
# - `ZtZmat`: The sum over j of Z_{(k,j)}'Z_{(k,j)}. This only need be 
#             calculated once so can be stored and re-entered for each
#             iteration.
#
# ============================================================================
def get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD, ZtZmat=None, reml=False, ZtX=None, XtiVX=None, ZtiVX=None):

  # Number of voxels
  v = Zte.shape[0]

  # Number of random effects
  q = ZtZ.shape[1]

  # Number of random factors
  r = len(nraneffs)

  if r == 1:
    # lk and qk for the first factor (zero indexed)
    l0 = nlevels[0]
    q0 = nraneffs[0]

  # We only need calculate this once across all iterations.
  if ZtZmat is None:

    # In the one factor one random effect setting this
    # computation boils down to a sum of square elements
    if r == 1 and nraneffs[0]==1:
      
      # We assume ZtZ is already diagonal
      ZtZmat = np.sum(ZtZ,axis=1).reshape((ZtZ.shape[0],1,1))

    # In the one factor multiple random effects setting this 
    # computation can also be simplified.
    elif r == 1 and nraneffs[0]>1:

      # We get can sum_j Z_(k,j)'Z_(k,j) by reshaping and summing
      # over an axis in this setting
      ZtZmat = np.sum(ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0],l0,q0,q0),axis=1)

    # In the general setting it is a sum of matrix products
    else:

      # Instantiate to zeros
      ZtZmat = np.zeros((ZtZ.shape[0],nraneffs[k],nraneffs[k]))

      for j in np.arange(nlevels[k]):

        # Get the indices for the kth factor jth level
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

        # Work out Z_(k,j)'Z_(k,j)
        ZtZterm = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)]

        # Add together
        ZtZmat = ZtZmat + ZtZterm

  # Get the indices for the factors 
  Ik = fac_indices2D(k, nlevels, nraneffs)

  # Work out lk
  lk = nlevels[k]

  # Work out block size and partition
  qk = nraneffs[k]
  pttn = np.array([qk,1])

  # We now work out the sum of Z_(k,j)'Z(I+Z'ZD)^(-1)Z'Z_(k,j). In the one random
  # factor, one random effect setting, this can be sped up massively using the
  # sumTTt_1fac1ran3D function.
  if r == 1 and nraneffs[0]==1:
    secondTerm = sumTTt_1fac1ran3D(ZtZ, DinvIplusZtZD, nlevels[k], nraneffs[k])
  
  # This can also be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0]>1:

    # Get blocks of Z'Z and D(I+Z'ZD)^{-1}
    ZtZblocks = ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0],l0,q0,q0)
    DinvIplusZtZDblocks = DinvIplusZtZD.transpose(0,2,1).reshape(v,l0,q0,q0)

    # Get second term
    secondTerm = np.sum(ZtZblocks @ DinvIplusZtZDblocks @ ZtZblocks,axis=1)

  else:
    # Work out the second term in TT'
    secondTerm = sumAijBijt3D(ZtZ[:,Ik,:] @ DinvIplusZtZD, ZtZ[:,Ik,:], pttn, pttn)

  # Obtain RkSum=sum (TkjTkj')
  RkSum = ZtZmat - secondTerm

  # Work out T_ku*sigma
  if r == 1 and nraneffs[0]==1:
    TuSig = Zte - np.einsum('ij,ij,ijk->ijk', ZtZ, DinvIplusZtZD, Zte)

  # This can be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0] > 1:

    # Get product of ZtZ and DinvIplusZtZD blocks
    ZtZDinvIplusZtZDZte = ZtZblocks @ DinvIplusZtZDblocks

    # Multiply by Zte
    ZtZDinvIplusZtZDZte = ZtZDinvIplusZtZDZte @ Zte.reshape(v,l0,q0,1)    

    # Reshape appropriately
    ZtZDinvIplusZtZDZte = ZtZDinvIplusZtZDZte.reshape(v,q0*l0,1)

    # Get TuSig
    TuSig = Zte - ZtZDinvIplusZtZDZte

  else:
    TuSig = Zte[:,Ik,:] - (ZtZ[:,Ik,:] @ (DinvIplusZtZD @ Zte))

  # Obtain Sum Tu(Tu)'
  TuuTSum = np.einsum('i,ijk->ijk',1/sigma2,sumAijBijt3D(TuSig, TuSig, pttn, pttn))

  # Work out dldDk
  dldDk = 0.5*(forceSym3D(TuuTSum - RkSum))

  if reml==True:

    if r == 1:

      # # Get Z'V^{-1}X
      # ZtiVX = ZtX - np.einsum('ij,ijk->ijk', ZtZ, np.einsum('ij,ijk->ijk',DinvIplusZtZD, ZtX))

      # Get q0 and p
      q0 = nraneffs[0]
      p = ZtiVX.shape[-1]

      # For ease, label A=Z'V^{-1}X and B=(X'V^{-1}X)^{-1}Z'V^{-1}X 
      A = ZtiVX
      Bt = np.linalg.inv(XtiVX) @ ZtiVX.transpose((0,2,1))

      # Peform vecm operation
      vecmAt = block2stacked3D(A.transpose((0,2,1)),[p,q0])
      vecmBt = block2stacked3D(Bt,[p,q0])

      # Update gradient
      dldDk = dldDk + 0.5*vecmAt.transpose((0,2,1)) @ vecmBt

    else:

      # Invert X'V^(-1)X
      iXtiVX = np.linalg.inv(XtiVX)

      # For each level j we need to add a term
      for j in np.arange(nlevels[k]):

        # Get the indices for the kth factor jth level
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

        Z_kjtZ = ZtZ[:,Ikj,:]
        Z_kjtX = ZtX[:,Ikj,:]

        Z_kjtinvVX = Z_kjtX - Z_kjtZ @ DinvIplusZtZD @ ZtX

        dldDk = dldDk + 0.5*Z_kjtinvVX @ iXtiVX @ Z_kjtinvVX.transpose((0,2,1))

  # Store it in the dictionary
  return(dldDk, ZtZmat)


# ============================================================================
#
# Commented out below is an older version of the above code. This has been 
# left here in case it has any use for future development.
#
# ============================================================================
# def get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,reml=False, ZtX=0, XtX=0):
#
#   # Number of voxels
#   v = Zte.shape[0]
#  
#   # Initalize the derivative to zeros
#   dldDk = np.zeros((v, nraneffs[k],nraneffs[k]))
#  
#   # For each level j we need to add a term
#   for j in np.arange(nlevels[k]):
#
#     # Get the indices for the kth factor jth level
#     Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
#    
#     # Get (the kj^th columns of Z)^T multiplied by Z
#     Z_kjtZ = ZtZ[:,Ikj,:]
#     Z_kjte = Zte[:,Ikj,:]
#    
#     # Get the first term of the derivative
#     Z_kjtVinve = Z_kjte - (Z_kjtZ @ DinvIplusZtZD @ Zte)
#     firstterm = np.einsum('i,ijk->ijk',1/sigma2,forceSym3D(Z_kjtVinve @ Z_kjtVinve.transpose((0,2,1))))
#    
#     # Get (the kj^th columns of Z)^T multiplied by (the kj^th columns of Z)
#     Z_kjtZ_kj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)]
#     secondterm = forceSym3D(Z_kjtZ_kj) - forceSym3D(Z_kjtZ @ DinvIplusZtZD @ Z_kjtZ.transpose((0,2,1)))
#    
#     if j == 0:
#      
#       # Start a running sum over j
#       dldDk = firstterm - secondterm
#      
#     else:
#    
#       # Add these to the running sum
#       dldDk = dldDk + firstterm - secondterm
#    
#   if reml==True:
#
#     invXtinvVX = np.linalg.inv(XtX - ZtX.transpose((0,2,1)) @ DinvIplusZtZD @ ZtX)
#
#     # For each level j we need to add a term
#     for j in np.arange(nlevels[k]):
#
#       # Get the indices for the kth factor jth level
#       Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
#
#       Z_kjtZ = ZtZ[:,Ikj,:]
#       Z_kjtX = ZtX[:,Ikj,:]
#
#       Z_kjtinvVX = Z_kjtX - Z_kjtZ @ DinvIplusZtZD @ ZtX
#
#       dldDk = dldDk + 0.5*Z_kjtinvVX @ invXtinvVX @ Z_kjtinvVX.transpose((0,2,1))
#
#   # Halve the sum (the coefficient of a half was not included in the above)
#   dldDk = forceSym3D(dldDk/2)
#
#   # Store it in the dictionary
#   return(dldDk)
# ============================================================================


# ============================================================================
#
# The below function calculates the covariance between the derivative of the 
# log likelihood with respect to \beta, given by the below formula:
#
# cov(dl/(d\beta)) = \sigma^(-2) X'(I+ZDZ')^(-1)X
#                  = \sigma^(-2) X'(I-ZD(I+Z'ZD)^(-1)Z')X
#                  = \sigma^(-2) (X'X-X'ZD(I+Z'ZD)^(-1)Z'X)
# 
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `XtZ`: X transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). 
# - `XtX`: X transpose multiplied by X (can be spatially varying or non
#          -spatially varying). 
# - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). If we are looking at a one random factor one
#          random effect design the variable ZtZ only holds the diagonal 
#          elements of the matrix Z'Z.
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `covdldbeta`: The covariance of the derivative of the log likelihood with 
#                 respect to \beta.
#
# ============================================================================
def get_covdldbeta3D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2, nraneffs):

  # Number of random factors
  r = len(nraneffs)

  # In the one random factor one random effect setting this computation can be 
  # streamlined
  if r == 1 and nraneffs[0]==1:

    # Get the covariance of the derivative
    covderiv = np.einsum('i,ijk->ijk',1/sigma2,(XtX - forceSym3D(XtZ @ np.einsum('ij,ijk->ijk', DinvIplusZtZD, XtZ.transpose((0,2,1))))))

  # This can also be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0] > 1:

    # Get p and q
    if XtZ.ndim==3:
      p = XtZ.shape[1]
      q = XtZ.shape[2]
    else:
      p = XtZ.shape[0]
      q = XtZ.shape[1]

    # Transpose
    ZtX = XtZ.transpose(0,2,1)

    # Get l0, q0 and p
    q0 = nraneffs[0]
    l0 = q//q0

    # Reshape DinvIplusZtZD appropriately
    DinvIplusZtZDZtX = DinvIplusZtZD.transpose(0,2,1).reshape(sigma2.shape[0],l0,q0,q0)

    # Multiply by ZtX
    DinvIplusZtZDZtX = DinvIplusZtZDZtX @ ZtX.reshape(ZtX.shape[0],l0,q0,p)    

    # Reshape appropriately
    DinvIplusZtZDZtX = DinvIplusZtZDZtX.reshape(sigma2.shape[0],q0*l0,p)

    # Get the covariance of the derivative
    covderiv = np.einsum('i,ijk->ijk',1/sigma2,(XtX - forceSym3D(XtZ @ DinvIplusZtZDZtX)))

  else:
    # Get the covariance of the derivative
    covderiv = np.einsum('i,ijk->ijk',1/sigma2,(XtX - forceSym3D(XtZ @ DinvIplusZtZD @ XtZ.transpose((0,2,1)))))

  # Return the covariance of the derivative
  return(covderiv)


# ============================================================================
#
# The below function calculates the covariance between the derivative of the 
# log likelihood with respect to vech(D_k) and the derivative with respect to 
# \sigma^2.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `k`: The number of the first factor (k in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and 
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). If we are looking at a one random factor one
#          random effect design the variable ZtZ only holds the diagonal 
#          elements of the matrix Z'Z.
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `dupMatTdict`: A dictionary of transpose duplication matrices such that 
#                   `dupMatTdict[k]` = DupMat_k'.
# - `vec`: This is a boolean value which by default is false. If True it gives
#          the update vector for vec (i.e. duplicates included), otherwise it
#          gives the update vector for vech.
# - `ZtZmat`: The sum over j of Z_{(k,j)}'Z_{(k,j)}. This only need be 
#             calculated once so can be stored and re-entered for each
#             iteration.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `covdldDdldsigma2`: The covariance between the derivative of the log 
#                       likelihood with respect to vech(D_k) and the 
#                       derivative with respect to \sigma^2.
# - `ZtZmat`: The sum over j of Z_{(k,j)}'Z_{(k,j)}. This only need be 
#             calculated once so can be stored and re-entered for each
#             iteration.
#
# ============================================================================
def get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=False, ZtZmat=None):

  # Number of voxels
  v = DinvIplusZtZD.shape[0]

  # Number of random factors r
  r = len(nraneffs)

  if r == 1:
    # lk and qk for the first factor (zero indexed)
    l0 = nlevels[0]
    q0 = nraneffs[0]

  # We only need calculate this once across all iterations
  if ZtZmat is None:

  # In the one random factor one random effect setting this computation can be 
  # streamlined
    if r == 1 and nraneffs[0]==1:
      
      # We assume ZtZ is already diagonal
      ZtZmat = np.sum(ZtZ,axis=1).reshape((ZtZ.shape[0],1,1))

    # In the one factor multiple random effects setting this 
    # computation can also be simplified.
    elif r == 1 and nraneffs[0]>1:

      # We get can sum_j Z_(k,j)'Z_(k,j) by reshaping and summing
      # over an axis in this setting
      ZtZmat = np.sum(ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0],l0,q0,q0),axis=1)

    # In the general setting it is a sum of matrix products
    else:

      # Instantiate to zeros
      ZtZmat = np.zeros((ZtZ.shape[0],nraneffs[k],nraneffs[k]))

      for j in np.arange(nlevels[k]):

        # Get the indices for the kth factor jth level
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

        # Work out Z_(k,j)'Z_(k,j)
        ZtZterm = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)]

        # Add together
        ZtZmat = ZtZmat + ZtZterm

  # Get the indices for the factors 
  Ik = fac_indices2D(k, nlevels, nraneffs)

  # Work out lk
  lk = nlevels[k]

  # Work out block size and partition
  q = np.sum(nlevels*nraneffs)
  qk = nraneffs[k]
  pttn = np.array([qk,q])

  # We now work out the sum of Z_(k,j)'ZD(I+Z'ZD)^(-1)ZZ_(k,j). In the one random
  # factor, one random effect setting, this can be sped up massively using the
  # sumTTt_1fac1ran3D function.
  if r == 1 and nraneffs[0]==1:
    secondTerm = sumTTt_1fac1ran3D(ZtZ, DinvIplusZtZD, nlevels[k], nraneffs[k])

  # This can also be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0]>1:

    # Get blocks of Z'Z and D(I+Z'ZD)^{-1}
    ZtZblocks = ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0],l0,q0,q0)
    DinvIplusZtZDblocks = DinvIplusZtZD.transpose(0,2,1).reshape(v,l0,q0,q0)

    # Get second term
    secondTerm = np.sum(ZtZblocks @ DinvIplusZtZDblocks @ ZtZblocks,axis=1)

  else:
    # Work out the second term
    secondTerm = sumAijBijt3D(ZtZ[:,Ik,:] @ DinvIplusZtZD, ZtZ[:,Ik,:], pttn, pttn)

  # Obtain ZtZmat
  RkSum = ZtZmat - secondTerm

  # Multiply by duplication matrices and half-vectorize/vectorize
  if not vec:
    covdldDdldsigma2 = np.einsum('i,ijk->ijk', 1/(2*sigma2), dupMatTdict[k] @ mat2vec3D(RkSum))
  else:
    covdldDdldsigma2 = np.einsum('i,ijk->ijk', 1/(2*sigma2), mat2vec3D(RkSum))

  return(covdldDdldsigma2, ZtZmat)


# ============================================================================
#
# Commented out below is an older version of the above code. This has been 
# left here in case it has any use for future development.
#
# ============================================================================
# def get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=False):
#  
#   # Number of voxels
#   v = DinvIplusZtZD.shape[0]
#  
#   # Sum of R_(k, j) over j
#   RkSum = np.zeros((v,nraneffs[k],nraneffs[k]))
#
#   for j in np.arange(nlevels[k]):
#
#     # Get the indices for the kth factor jth level
#     Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
#
#     # Work out R_(k, j)
#     Rkj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)] - forceSym3D(ZtZ[:,Ikj,:] @ DinvIplusZtZD @ ZtZ[:,:,Ikj])
#    
#     # Add together
#     RkSum = RkSum + Rkj
#
#   # Multiply by duplication matrices and 
#   if not vec:
#     covdldDdldsigma2 = np.einsum('i,ijk->ijk', 1/(2*sigma2), dupMatTdict[k] @ mat2vec3D(RkSum))
#   else:
#     covdldDdldsigma2 = np.einsum('i,ijk->ijk', 1/(2*sigma2), mat2vec3D(RkSum))
#  
#   return(covdldDdldsigma2)
# ============================================================================


# ============================================================================
#
# The below function calculates the covariance between the derivative of the 
# log likelihood with respect to vech(D_(k1)) and the derivative with respect 
# to vech(D_(k2)).
#
# cov(dl/(dvech(D_(k1))),dl/(dvech(D_(k2))))=
#      0.5DupMat_(k1)^+\sum_(j=1)^(l_(k2))\sum_(j=1)^(l_(k1))(R_(k1,k2,i,j) kron ...
#      R_(k1,k2, i,j))DupMat_(k2)^+'
#
# Where R_(k1,k2,i,j)=Z_(k1,i)'(I+ZDZ')^(-1)Z_(k2,j)=Z_(k1,i)'Z_(k2,j) - ...
# Z_(k1,i)'ZD(I+Z'ZD)^(-1)Z_(k2,j)..
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `k1`: The number of the first factor (k1 in the above notation).
# - `k2`: The number of the second factor (k2 in the above notation).
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `ZtZ`: Z transpose multiplied by Z (can be spatially varying or non
#          -spatially varying). If we are looking at a one random factor one
#          random effect design the variable ZtZ only holds the diagonal 
#          elements of the matrix Z'Z.
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1).
# - `dupMatTdict`: A dictionary of transpose duplication matrices such that 
#                    `dupMatTdict[k]` = DupMat_k'
# - `vec`: This is a boolean value which by default is false. If True it gives
#          the update matrix for vec (i.e. duplicates included), otherwise it
#          gives true covariance for vech.
# - `perm` (optional): The permutation of I kron K kron I (see 
#                      `permOfIkKkI2D`). This only need be calculated once so
#                      can be passed between iterations.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `covdldDk1dldk2`: The covariance between the derivative of the log
#                     likelihood with respect to vech(D_(k1)) and the 
#                     derivative with respect to vech(D_(k2)).
# - `perm`: The permutation of I kron K kron I (see `permOfIkKkI2D`). This 
#           only need be calculated once so can be passed between iterations.
#
# ============================================================================
def get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=None, vec=False):
  
  # Get the indices for the factors 
  Ik1 = fac_indices2D(k1, nlevels, nraneffs)
  Ik2 = fac_indices2D(k2, nlevels, nraneffs)

  # Work out number of random factors
  r = len(nlevels)

  if r == 1:

    # Shorthand q0 and l0
    q0 = nraneffs[0]
    l0 = nlevels[0]

  # Work out number of voxels
  v = DinvIplusZtZD.shape[0]

  # Work out R_(k1,k2) (in the one factor, one random effect setting we can speed this up a lot)
  if r == 1 and nraneffs[0] == 1:

    # Rk1k2 diag (we assume in this setting that ZtZ and DinvIplusZtZD only contain the diagonal
    # elements here)
    Rk1k2diag = ZtZ - DinvIplusZtZD*ZtZ**2

    # lk and qk for the first factor (zero indexed)
    l0 = nlevels[0]
    q0 = nraneffs[0]

    # Get diagonal values of R and sum the squares. This is equivalent
    # to the kron operation in the one factor case
    RkRSum = np.sum(Rk1k2diag**2,axis=1).reshape(v, q0**2, q0**2)

    # This is the covariance
    covdldDk1dldk2 = 1/2*RkRSum

  # This can also be performed faster in the one factor, mutliple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD. 
  elif r == 1 and nraneffs[0] > 1:

    # Get blocks of Z'Z and D(I+Z'ZD)^{-1}
    ZtZblocks = ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0],l0,q0,q0)
    DinvIplusZtZDblocks = DinvIplusZtZD.transpose(0,2,1).reshape(v,l0,q0,q0)

    # Get second term
    secondTerm = ZtZblocks @ DinvIplusZtZDblocks @ ZtZblocks

    # Obtain RkSum=sum (TkjTkj')
    Rk = ZtZ.transpose(0,2,1).reshape(ZtZ.shape[0],l0,q0,q0) - secondTerm

    # Get new kronecker
    RkRSum = np.sum(kron4D(Rk,Rk),axis=1)

    # This is the covariance
    covdldDk1dldk2 = 1/2*RkRSum

    # Multiply by duplication matrices and save
    if not vec:
      covdldDk1dldk2 = 1/2 * dupMatTdict[k1] @ RkRSum @ dupMatTdict[k2].transpose()
    else:
      covdldDk1dldk2 = 1/2 * RkRSum

  else:

    # Get R_(k1,k2)=Z'V^(-1)Z_(k1,k2)
    Rk1k2 = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ik1,Ik2)] - (ZtZ[:,Ik1,:] @ DinvIplusZtZD @ ZtZ[:,:,Ik2])
    
    # Work out block sizes
    pttn = np.array([nraneffs[k1],nraneffs[k2]])

    # Obtain permutation
    RkRSum,perm=sumAijKronBij3D(Rk1k2, Rk1k2, pttn, perm)

    # Multiply by duplication matrices and save
    if not vec:
      covdldDk1dldk2 = 1/2 * dupMatTdict[k1] @ RkRSum @ dupMatTdict[k2].transpose()
    else:
      covdldDk1dldk2 = 1/2 * RkRSum

  # Return the result
  return(covdldDk1dldk2, perm)


# ============================================================================
#
# Commented out below is an older version of the above code. This has been 
# left here in case it has any use for future development.
#
# ============================================================================
# def get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=False):
#  
#   # Sum of R_(k1, k2, i, j) kron R_(k1, k2, i, j) over i and j 
#   for i in np.arange(nlevels[k1]):
#
#     for j in np.arange(nlevels[k2]):
#      
#       # Get the indices for the k1th factor jth level
#       Ik1i = faclev_indices2D(k1, i, nlevels, nraneffs)
#       Ik2j = faclev_indices2D(k2, j, nlevels, nraneffs)
#      
#       # Work out R_(k1, k2, i, j)
#       Rk1k2ij = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ik1i,Ik2j)] - (ZtZ[:,Ik1i,:] @ DinvIplusZtZD @ ZtZ[:,:,Ik2j])
#     
#       # Work out Rk1k2ij kron Rk1k2ij
#       RkRt = kron3D(Rk1k2ij,Rk1k2ij)
#      
#       # Add together
#       if (i == 0) and (j == 0):
#      
#         RkRtSum = RkRt
#      
#       else:
#        
#         RkRtSum = RkRtSum + RkRt
#    
#   # Multiply by duplication matrices and save
#   if not vec:
#     covdldDk1dldk2 = 1/2 * dupMatTdict[k1] @ RkRtSum @ dupMatTdict[k2].transpose()
#   else:
#     covdldDk1dldk2 = 1/2 * RkRtSum
#
#   # Return the result
#   return(covdldDk1dldk2)
# ============================================================================


# ============================================================================
#
# This function is used to generate indices for converged and non-converged
# voxels, before, after and during the current iteration of an algorithm.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `convergedBeforeIt`: Boolean array of all voxels which had converged
#                        before the iteration, relative to the full list of 
#                        voxels. 
# - `convergedDuringIt`: Boolean array of all voxels which converged during
#                        the iteration, relative to the list of voxels
#                        considered during the iteration.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `indices_ConAfterIt`: The indices of all voxels converged after the 
#                         iteration, relative to the full list of voxels. 
# - `indices_notConAfterIt`: The indices of all voxels not converged after the
#                            iteration, relative to the full list of voxels.
# - `indices_conDuringIt`: The indices of all voxels which converged during 
#                          the iteration, relative to the full list of voxels.
# - `local_converged`: The indices of the voxels which converged during the
#                      iteration, relative to the list of voxels considered
#                      during the iteration.
# - `local_notconverged`:The indices of the voxels did not converge during the
#                        iteration, and were not converged prior to the 
#                        iteration, relative to the list of voxels considered
#                        during the iteration.
#
# ----------------------------------------------------------------------------
#
# Developer note: If this documentation seems confusing, see the example in 
# `unitTests3D.py`. It might help shed some light on what's going on here.
#
# ============================================================================
def getConvergedIndices(convergedBeforeIt, convergedDuringIt):
  
  # ==========================================================================
  # Global indices (i.e. relative to whole image)
  # --------------------------------------------------------------------------
  
  # Numbers 1 to number of voxels
  indices = np.arange(len(convergedBeforeIt))
  
  # Indices of those which weren't converged before the iteration
  indices_notConBeforeIt = indices[convergedBeforeIt==0]
  
  # Indices of those which weren't converged after the iteration
  indices_notConAfterIt = indices_notConBeforeIt[convergedDuringIt==0]
  
  # Indices of those which were converged after iteration
  indices_ConAfterIt = np.setdiff1d(indices, indices_notConAfterIt)
  
  # Indices of those which converged during iteration
  indices_conDuringIt = np.setdiff1d(indices_notConBeforeIt, indices_notConAfterIt)
  
  # ==========================================================================
  # Local indices (i.e. relative to current voxels)
  # --------------------------------------------------------------------------
  local_converged = np.arange(len(convergedDuringIt))[convergedDuringIt==1]
  local_notconverged = np.arange(len(convergedDuringIt))[convergedDuringIt==0]
  
  return(indices_ConAfterIt, indices_notConAfterIt, indices_conDuringIt, local_converged, local_notconverged)


# ============================================================================
# 
# This function converts a 3D matrix partitioned into blocks into a 3D matrix 
# with each 2D submatrix consisting of the blocks stacked on top of one 
# another. I.e. for each matrix Ai=A[i,:,:], it maps Ai to matrix Ai_s like
# so:
#
#                                                           |   Ai_{1,1}   |
#                                                           |   Ai_{1,2}   |
#      | Ai_{1,1}    Ai_{1,2}  ...  Ai_{1,l_2}  |           |     ...      |
#      | Ai_{2,1}    Ai_{2,2}  ...  Ai_{2,l_2}  |           |  Ai_{1,l_2}  |
# Ai = |    ...        ...     ...       ...    | -> Ai_s = |   Ai_{2,1}   |
#      | Ai_{l_1,1} Ai_{l_1,2} ... Ai_{l_1,l_2} |           |     ...      |
#                                                           |     ...      |
#                                                           | Ai_{l_1,l_2} |
#
# ----------------------------------------------------------------------------
#
# This function takes as inputs:
# 
# ----------------------------------------------------------------------------
#
#  - A: A 3D matrix of dimension (v by m1 by m2).
#  - pA: The size of the block partitions of the Ai, e.g. if A_{i,j} is of 
#        dimension (n1 by n2) then pA=[n1, n2].
# 
# ----------------------------------------------------------------------------
#
# And returns as output:
#
# ----------------------------------------------------------------------------
#
#  - As: The matrix A reshaped to have for each i all blocks Ai_{i,j} on top
#        of one another. I.e. the above mapping has been performed.
#
# ============================================================================
def block2stacked3D(A, pA):

  # Work out shape of A
  v = A.shape[0] # (Number of voxels)
  m1 = A.shape[1]
  m2 = A.shape[2]

  # Work out shape of As
  n1 = pA[0]
  n2 = pA[1]
  
  # Change A to stacked form
  As = A.reshape((v,m1//n1,n1,m2//n2,n2)).transpose(0,1,3,2,4).reshape(v,m1*m2//n2,n2)

  return(As)


# ============================================================================
# 
# This function converts a 3D matrix partitioned into blocks into a 3D matrix 
# with each 2D submatrix consisting of the blocks converted to vectors stacked
# on top of one another. I.e. for each matrix Ai=A[i,:,:], it maps Ai to
# matrix Ai_s like so:
#
#                                                               |   vec'(Ai_{1,1})   |
#                                                               |   vec'(Ai_{1,2})   |
#      | Ai_{1,1}    Ai_{1,2}  ...  Ai_{1,l_2}  |               |        ...         |
#      | Ai_{2,1}    Ai_{2,2}  ...  Ai_{2,l_2}  |               |  vec'(Ai_{1,l_2})  |
# Ai = |    ...         ...    ...      ...     | -> vecb(Ai) = |   vec'(Ai_{2,1})   |
#      | Ai_{l_1,1} Ai_{l_1,2} ... Ai_{l_1,l_2} |               |        ...         |
#                                                               |        ...         |
#                                                               | vec'(Ai_{l_1,l_2}) |
#
# ----------------------------------------------------------------------------
#
# The below function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - mat: A 3D matrix of dimension (v by m1 by m2).
#  - pA: The size of the block partitions of the mat_i, e.g. if Ai_{i,j} is of 
#        dimension (n1 by n2) then pA=[n1, n2].
#
# ----------------------------------------------------------------------------
#
# And gives the following outputs:
#
# ----------------------------------------------------------------------------
#
#  - vecb: A matrix composed of each block of mat, converted to row vectors, 
#          stacked on top of one another. I.e. for an arbitrary matrix A of 
#          appropriate dimensions, vecb(A) is the result of the above mapping,
#          where Ai_{j,k} has dimensions (1 by p[0] by p[1]) for all i, j and
#          k.
#
# ============================================================================
def mat2vecb3D(mat,p):

  # Change to stacked block format, if necessary
  if p[1]!=mat.shape[2]:
    mat = block2stacked3D(mat,p)

  # Get height of block.
  n = p[0]
  
  # Work out shape of matrix.
  v = mat.shape[0]
  m = mat.shape[1]
  k = mat.shape[2]

  # Convert to stacked vector format
  vecb = mat.reshape(v,m//n, n, k).transpose((0,2, 1, 3)).reshape(v,n, m*k//n).transpose((0,2,1)).reshape(v,m//n,n*k)

  #Return vecb
  return(vecb)


# ============================================================================
#
# The below function computes, given two 3D matrices A and B, and denoting Av
# and Bv as A[v,:,:] and B[v,:,:], the below sum, for all v:
#
#                 S = Sum_i Sum_j (Av_{i,j}Bv_{i,j}')
# 
# where the matrices A and B are block partitioned like so:
#
#      |   Av_{1,1}  ...  Av_{1,l2}  |       |   Bv_{1,1}  ...  Bv_{1,l2}  | 
# Av = |    ...      ...      ...    |  Bv = |     ...     ...     ...     | 
#      |  Av_{l1,1}  ...  Av_{l1,l2} |       |  Bv_{l1,1}  ...  Bv_{l1,l2} | 
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - A: A 3D matrix of dimension (v1* by m1 by m2).
#  - B: A 3D matrix of dimension (v2* by m1' by m2).
#  - pA: The size of the block partitions of A, e.g. if Av_{i,j} is of 
#        dimension (n1 by n2) then pA=[n1, n2].
#  - pB: The size of the block partitions of B, e.g. if Bv_{i,j} is of 
#        dimension (n1' by n2) the pB=[n1', n2].
#
# * v1 and v2 may differ if and only if one of them is set to 1 (i.e. we allow
#   for v1 and v2 to differ in the case that one of A or B is spatially
#   varying whilst the other is not).
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
#  - S: The sum of the partitions of Av multiplied by the transpose of the 
#       partitions of Bv, for every v.
# 
# ----------------------------------------------------------------------------
#
# Developer note: Note that the above implies that l1 must equal m1/n1=m1'/n1'
#                 and l2=m2/n2.
#
# ============================================================================
def sumAijBijt3D(A, B, pA, pB):
  
  # Number of voxels (we allow v1 and v2 to be different to allow for the 
  # case that one of A and B is not spatially varying and hence had v=1)
  v1 = A.shape[0]
  v2 = B.shape[0]

  # Work out second (the common) dimension of the reshaped A and B
  nA = pA[0]
  nB = pB[0]

  # Work out the first (the common) dimension of reshaped A and B
  mA = A.shape[1]*A.shape[2]//nA
  mB = B.shape[1]*B.shape[2]//nB

  # Check mA equals mB
  if mA != mB:
    raise Exception('Matrix dimensions incompatible.')

  # Convert both matrices to stacked block format.
  A = block2stacked3D(A,pA)
  B = block2stacked3D(B,pB)

  # Work out the sum
  S = A.transpose((0,2,1)).reshape((v1,mA,nA)).transpose((0,2,1)) @ B.transpose((0,2,1)).reshape((v2,mB,nB))

  # Return result
  return(S)


# ============================================================================
#
# The below function computes the sum of T_(k,j)'T_(k,j) across all levels j 
# where T_(k,j)=Z_(k,j)'ZD(I+Z'ZD)^(-1/2), for the one random effect, one
# random factor setting.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). As we are looking at a 
#                    single random factor single random effect model,  
#                    DinvIplusZtZD must only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1).
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation). As
#          we are looking at a one random factor, one random effect design 
#          the variable ZtZ must only hold the diagonal elements of the matrix
#          Z'Z.
# - `l0`: The number of levels for the one random factor.
# - `q0`: The number of random effects (should be 1).
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
#  - `sumTTt`: The sum of Z_(k,j)'ZD(I+Z'ZD)^(-1)Z'Z_(k,j) across all levels j. 
#
# ============================================================================
def sumTTt_1fac1ran3D(ZtZ, DinvIplusZtZD, l0, q0):

  # Number of voxels, v
  v = DinvIplusZtZD.shape[0]

  # Work out the diagonal values of the matrix product Z'ZD(I+Z'ZD)^(-1)Z'Z
  DiagVals = DinvIplusZtZD*ZtZ**2

  # Reshape diag vals and sum apropriately
  DiagVals = np.sum(DiagVals.reshape(v,q0,l0),axis=2)

  # Put values back into a matrix
  sumTTt = np.zeros((v,q0,q0))
  np.einsum('ijj->ij', sumTTt)[...] = DiagVals

  return(sumTTt)


# =============================================================================
#
# The below function is designed for use in the case when the LMM contains one
# random factor but multiple random effects. In this case, it takes ZtZ, which
# is assumed to be block diagonal and ``flattens" it (i.e. it removes all zeros
# and retains only the blocks on the main diagonal as a horizontal matrix. I.e.
# for a block diagonal ZtZ it performs a mapping like the below:
#
#         | A 0 0 |
#         | 0 B 0 | -->  | A B C |
#         | 0 0 C |
#
# -----------------------------------------------------------------------------
#
# It takes the following inputs:
#
# -----------------------------------------------------------------------------
#
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation). As
#          we are looking at a one random factor, multiple random effecs
#          ZtZ is assumed to be block diagonal.
# - `l0`: The number of levels for the one random factor.
# - `q0`: The number of random effects.
#
# -----------------------------------------------------------------------------
#
# And gives the following outputs:
#
# -----------------------------------------------------------------------------
#
#   - `ZtZ_flattened`: The matrix ZtZ `flattened` from shape (qxq) to shape
#                      (q_0xq).
#
# -----------------------------------------------------------------------------
def flattenZtZ(ZtZ, l0, q0):

  # Get q
  q = ZtZ.shape[-1]

  # In the case ZtZ is 2D we don't worry about voxel index
  if ZtZ.ndim==2:

    # Flatten ZtZ
    ZtZ_flattened = np.sum(ZtZ.reshape(l0,q0,q),axis=0)

  # Else ZtZ is assumed 3D
  else:

    # Get voxel number
    v = ZtZ.shape[0]

    # Flatten ZtZ_sv
    ZtZ_flattened = np.sum(ZtZ.reshape(v,l0,q0,q),axis=1)

  return(ZtZ_flattened)


# ============================================================================
#
# The below function computes, given two 3D matrices A and B, and denoting Av
# and Bv as A[v,:,:] and B[v,:,:], the below sum, for all v:
#
#                 S = Sum_i Sum_j (Av_{i,j} kron Bv_{i,j})
# 
# where the matrices A and B are block partitioned like so:
#
#      |   Av_{1,1}  ...  Av_{1,l2}  |       |   Bv_{1,1}  ...  Bv_{1,l2}  | 
# Av = |    ...      ...      ...    |  Bv = |     ...     ...     ...     | 
#      |  Av_{l1,1}  ...  Av_{l1,l2} |       |  Bv_{l1,1}  ...  Bv_{l1,l2} | 
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - `A`: A 3D matrix of dimension (v by m1 by m2).
#  - `B`: A 3D matrix of dimension (v by m1 by m2).
#  - `pttn`: The size of the block partitions of A and B, e.g. if A_{i,j} and 
#            B_{i,j} are of dimension (n1 by n2) then pA=[n1, n2].
#  - `perm` (optional): The permutation vector representing the matrix kronecker
#                       product I_{n2} kron K_{n2,n1} kron I_{n1}.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `S`: The sum of the partitions of Av multiplied by the transpose of the 
#        partitions of Bv, for every v; i.e. the sum given above.
# - `perm`: The permutation (same as input) used for calculation (useful for 
#           later computation).
#
# ============================================================================
def sumAijKronBij3D(A, B, pttn, perm=None):

  # Check dim A and B and pA and pB all same
  n1 = pttn[0]
  n2 = pttn[1]

  # Number of voxels
  v = A.shape[0]

  # This matrix only needs be calculated once
  if perm is None:
    perm = permOfIkKkI2D(n2,n1,n2,n1) 

  # Convert to vecb format
  atilde = mat2vecb3D(A,pttn)
  btilde = mat2vecb3D(B,pttn)

  # Multiply and convert to vector
  vecba = mat2vec3D(btilde.transpose((0,2,1)) @ atilde)

  # Permute
  S_noreshape = vecba[:,perm,:] 

  # Reshape to correct shape
  S = S_noreshape.reshape(v,n2**2,n1**2).transpose((0,2,1))

  return(S,perm)


# ============================================================================
#
# The below function calculates the residual mean squares for a beta estimate
# give by:
#
#   resms = (Y-X\beta)'(Y-X\beta)/(n-p)
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `YtX`: Y transpose multiplied by X (Y'X in the above notation).
# - `YtY`: Y transpose multiplied by Y (Y'Y in the above notation).
# - `XtX`: X transpose multiplied by X (X'X in the above notation).
# - `beta`: An estimate of the parameter vector (\beta in the above notation).
# - `n`: The number of observations/input niftis (potentially spatially
#        varying)
# - `p`: The number of fixed effects parameters.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `resms`: The residual mean squares.
#
# ============================================================================
def get_resms3D(YtX, YtY, XtX, beta, n, p):

    ete = ssr3D(YtX, YtY, XtX, beta)

    # Reshape n if necessary
    if isinstance(n,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(n.shape)>1:
    
            n = n.reshape(ete.shape)

    return(ete/(n-p))


# ============================================================================
#
# The below function gives the covariance matrix of the beta estimates.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `XtiVX`: The matrix X'V^{-1}X.
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `covB`: The covariance of the beta estimates.
#
# ============================================================================
def get_covB3D(XtiVX, sigma2, nraneffs):

    # Number of random factors r
    r = len(nraneffs)

    # Reshape n if necessary
    if isinstance(sigma2,np.ndarray):

        # Check first that n isn't a single value
        if sigma2.ndim>1:
          
            sigma2 = sigma2.reshape(sigma2.shape[0])

    # Work out cov(B)
    covB = np.linalg.inv(XtiVX)

    # Calculate sigma^2(X'V^{-1}X)^(-1)
    covB = np.einsum('i,ijk->ijk',sigma2,covB)

    # Return result
    return(covB)


# ============================================================================
#
# The below function calculates the (in most applications, scalar) variance
# of L\beta.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast vector (L can also be a matrix, but this isn't often the
#        case in practice when using this function).
# - `XtiVX`: The matrix X'V^{-1}X.
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `varLB`: The (usually scalar) variance of L\beta.
#
# ============================================================================
def get_varLB3D(L, XtiVX, sigma2, nraneffs):

    # Reshape n if necessary
    if isinstance(sigma2,np.ndarray):

        # Check first that n isn't a single value
        if sigma2.ndim>1:
    
            sigma2 = sigma2.reshape(sigma2.shape[0])

    # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
    varLB = L @ get_covB3D(XtiVX, sigma2, nraneffs) @ L.transpose()

    # Return result
    return(varLB)


# ============================================================================
#
# The below function calculates the partial R^2 statistic given, in terms of
# an F statistic by:
#
#    R^2 = df1*F/(df1*F+df2)
#
# Where df1 and df2 and the numerator and denominator degrees of freedom of
# F respectively.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast matrix.
# - `F`: A matrix of F statistics.
# - `df`: The denominator degrees of freedom of the F statistic (can be 
#         spatially varying).
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `R2`: A matrix of R^2 statistics.
#
# ============================================================================
def get_R23D(L, F, df):

    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Convert F to R2
    R2 = (rL*F)/(rL*F + df)

    # Return R2
    return(R2)


# ============================================================================
#
# The below function calculates the approximate T statistic for a null
# hypothesis test, H0:L\beta == 0 vs H1: L\beta != 0. The T statistic is given
# by:
#
#     T = L\beta/s.e.(L\beta)
#
# Where s.e. represents standard error.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast vector.
# - `XtiVX`: The matrix X'V^{-1}X.
# - `beta`: The estimate of the fixed effects parameters.
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `T`: A matrix of T statistics.
#
# ============================================================================
def get_T3D(L, XtiVX, beta, sigma2, nraneffs):

    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Work out Lbeta
    LB = L @ beta

    # Work out se(T)
    varLB = get_varLB3D(L, XtiVX, sigma2, nraneffs)

    # Work out T
    T = LB/np.sqrt(varLB)

    # Return T
    return(T)


# ============================================================================
#
# The below function calculates the approximate F staistic given by:
#
#    F = (L\beta)'(L(X'V^(-1)X)^(-1)L')^(-1)(L\beta)/rank(L)
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast matrix.
# - `XtiVX`: The matrix X'V^{-1}X.
# - `beta`: The estimate of the fixed effects parameters.
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `F`: A matrix of F statistics.
#
# ============================================================================
def get_F3D(L, XtiVX, betahat, sigma2, nraneffs):

    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Work out Lbeta
    LB = L @ betahat

    # Work out se(F)
    varLB = get_varLB3D(L, XtiVX, sigma2, nraneffs)

    # Work out F
    F = LB.transpose(0,2,1) @ np.linalg.inv(varLB) @ LB/rL

    # Return T
    return(F)


# ============================================================================
#
# The below function converts T statistics to -log10(P) values. `-inf` values
# are replace by minlog.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `T`: A matrix of T statistics.
# - `df`: The degrees of freedom of the T statistic (can be spatially varying).
# - `minlog`: A value to replace `-inf` p-values with.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `P`: The matrix of -log10(p) values.
#
# ============================================================================
def T2P3D(T,df,minlog):

    # Initialize empty P
    P = np.zeros(np.shape(T))

    # Do this seperately for >0 and <0 to avoid underflow
    P[T < 0] = -np.log10(1-stats.t.cdf(T[T < 0], df[T < 0]))
    P[T >= 0] = -np.log10(stats.t.cdf(-T[T >= 0], df[T >= 0]))

    # Remove infs
    P[np.logical_and(np.isinf(P), P<0)]=minlog

    return(P)


# ============================================================================
#
# The below function converts F statistics to -log10(P) values. `-inf` values
# are replace by minlog.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `F`: A matrix of F statistics.
# - `L`: A contrast matrix.
# - `df_denom`: The denominator degrees of freedom of the F statistic (can be 
#               spatially varying).
# - `minlog`: A value to replace `-inf` p-values with.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `P`: The matrix of -log10(p) values.
#
# ============================================================================
def F2P3D(F, L, df_denom, minlog):

    # Get the rank of L
    df_num = np.linalg.matrix_rank(L)

    # Work out P
    P = -np.log10(1-stats.f.cdf(F, df_num, df_denom))

    # Remove infs
    P[np.logical_and(np.isinf(P), P<0)]=minlog

    return(P)


# ============================================================================
#
# The below function estimates the degrees of freedom for an F statistic using
# a Sattherthwaite approximation method. For, a contrast matrix L, this 
# estimate is given by:
#
#      v = (sum_{i=0}^rank(L) v_{l_i})/((sum_{i=0}^rank(L) v_{l_i}) - rank(L))
#
# Where l_i is the i^th row of L and v_{l_i} is the sattherthwaithe estimate
# of the degrees of freedom of a T statistic with contrast l_i (see 
# `get_swdf_T3D` below). 
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast matrix.
# - `sigma2`: The fixed effects variance estimate.
# - `XtiVX`: The matrix X'V^{-1}X.
# - `ZtiVX`: The matrix Z'V^{-1}X.
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `ZtX`: Z transpose multiplied by X (Z'X in the previous notation).
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation). If 
#          we are looking at a one random factor one random effect design 
#          the variable ZtZ only holds the diagonal elements of the matrix 
#          Z'Z.
# - `n`: The number of observations/input niftis (potentially spatially
#        varying)
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `df`: The spatially varying Sattherthwaithe degrees of freedom estimate.
#
# ============================================================================
def get_swdf_F3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ, DinvIplusZtZD, n, nlevels, nraneffs): 

    # Reshape sigma2 if necessary
    sigma2 = sigma2.reshape(sigma2.shape[0])

    # Reshape n if necessary
    if isinstance(n,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(n.shape)>1:
    
            n = n.reshape(sigma2.shape)

    # L is rL in rank
    rL = np.linalg.matrix_rank(L)

    # Initialize empty sum.
    sum_swdf_adj = np.zeros(sigma2.shape)

    # Loop through first rL rows of L
    for i in np.arange(rL):

        # Work out the swdf for each row of L
        swdf_row = get_swdf_T3D(L[i:(i+1),:], sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ, DinvIplusZtZD, n, nlevels, nraneffs)

        # Work out adjusted df = df/(df-2)
        swdf_adj = swdf_row/(swdf_row-2)

        # Add to running sum
        sum_swdf_adj = sum_swdf_adj + swdf_adj.reshape(sum_swdf_adj.shape)

    # Work out final df
    df = 2*sum_swdf_adj/(sum_swdf_adj-rL)

    # Return df
    return(df)


# ============================================================================
#
# The below function estimates the degrees of freedom for an T statistic using
# a Sattherthwaite approximation method. For, a contrast matrix L, this 
# estimate is given by:
#
#    v = 2(Var(L\beta)^2)/(d'I^{-1}d)
#
# Where d is the derivative of Var(L\beta) with respect to the variance 
# parameter vector \theta = (\sigma^2, vech(D_1),..., vech(D_r)) and I is the
# Fisher Information matrix of \theta.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast vector.
# - `sigma2`: The fixed effects variance estimate.
# - `XtiVX`: The matrix X'V^{-1}X.
# - `ZtiVX`: The matrix Z'V^{-1}X.
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `ZtX`: Z transpose multiplied by X (Z'X in the previous notation).
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation). If 
#          we are looking at a one random factor one random effect design 
#          the variable ZtZ only holds the diagonal elements of the matrix 
#          Z'Z.
# - `n`: The number of observations/input niftis (potentially spatially
#        varying)
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `df`: The spatially varying Sattherthwaithe degrees of freedom estimate.
#
# ============================================================================
def get_swdf_T3D(L, sigma2, XtiVX, ZtiVX, XtZ, ZtX, ZtZ, DinvIplusZtZD, n, nlevels, nraneffs): 

    # Reshape sigma2 if necessary
    sigma2 = sigma2.reshape(sigma2.shape[0])

    # Reshape n if necessary
    if isinstance(n,np.ndarray):

        # Check first that n isn't a single value
        if np.prod(n.shape)>1:
    
            n = n.reshape(sigma2.shape)

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB3D(L, XtiVX, sigma2, nraneffs)
    
    # Get derivative of S^2
    dS2 = get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2)

    # Get Fisher information matrix
    InfoMat = get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ)

    # Calculate df estimator
    df = 2*(S2**2)/(dS2.transpose(0,2,1) @ np.linalg.solve(InfoMat, dS2))

    # Return df
    return(df)


# ============================================================================
#
# The below function calculates the derivative of Var(L\beta) with respect to
# the variance parameter vector \theta = (\sigma^2, vech(D_1),..., vech(D_r)).
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `L`: A contrast vector.
# - `XtiVX`: The matrix X'V^{-1}X.
# - `ZtiVX`: The matrix Z'V^{-1}X.
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `sigma2`: The fixed effects variance estimate.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `dS2`: The derivative of var(L\beta) with respect to \theta.
#
# ============================================================================
def get_dS23D(nraneffs, nlevels, L, XtiVX, ZtiVX, sigma2):

  # Number of random effects, r
  r = len(nraneffs)

  # Number of voxels
  v = XtiVX.shape[0]

  # Number of fixed effects p
  p = XtiVX.shape[-1]

  # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
  dS2 = np.zeros((v, 1+np.int32(np.sum(nraneffs*(nraneffs+1)/2)),1))

  # Work out indices for each start of each component of vector 
  # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
  DerivInds = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
  DerivInds = np.insert(DerivInds,0,1)

  # Work of derivative wrt to sigma^2
  dS2dsigma2 = L @ np.linalg.pinv(XtiVX) @ L.transpose()

  # Add to dS2
  dS2[:,0:1] = dS2dsigma2.reshape(dS2[:,0:1].shape)

  # Work out T_ku*sigma. In the one random factor one random effect setting 
  # this computation is much quicker as we already have DinvIplusZtZD and ZtZ
  # in diagonal form
  if r == 1 and nraneffs[0]==1:

    # Obtain ZtX(XtiVX)^(-1)L'
    ZtiVXinvXtiVXLt = ZtiVX @ (np.linalg.inv(XtiVX) @ L.transpose())

    # Get the squared elements of Z'X(X'V^(-1)X)^(-1)L'. These are the terms in the
    # sum of the kronecker product.
    kronTerms = (ZtiVXinvXtiVXLt)**2

    # Get the derivative by summing the kronecker product terms
    dS2dvechDk = np.einsum('i,ij->ij',sigma2, np.sum(kronTerms, axis=1)).reshape((v,1,1)) 

    # Add to dS2
    dS2[:,DerivInds[0]:DerivInds[1]] = dS2dvechDk.reshape(dS2[:,DerivInds[0]:DerivInds[1]].shape)

  # This can also be performed faster in the one factor, multiple random effect
  # case by using only the diagonal blocks of DinvIplusZtZD 
  elif r == 1 and nraneffs[0] > 1:

    # Get (X'V^{-1}X)^{-1}L'
    iXtiVXLt = np.linalg.inv(XtiVX) @ L.transpose()

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

      # Initialize an empty zeros matrix
      dS2dvechDk = np.zeros((np.int32(nraneffs[k]*(nraneffs[k]+1)/2),1))#...

      for j in np.arange(nlevels[k]):

        # Get the indices for this level and factor.
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
                
        # Work out Z_(k,j)'V^{-1}X
        ZkjtiVX = ZtiVX[:,Ikj,:]

        # Work out the term to put into the kronecker product
        # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
        K = ZkjtiVX @ iXtiVXLt
        
        # Sum terms
        dS2dvechDk = dS2dvechDk + dupMat2D(nraneffs[k]).toarray().transpose() @ mat2vec3D(kron3D(K,K.transpose(0,2,1)))

      # Multiply by sigma^2
      dS2dvechDk = np.einsum('i,ijk->ijk',sigma2,dS2dvechDk)

      # Add to dS2
      dS2[:,DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2[:,DerivInds[k]:DerivInds[k+1]].shape)

  else:

    # Get (X'V^{-1}X)^{-1}L'
    iXtiVXLt = np.linalg.inv(XtiVX) @ L.transpose()

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

      # Initialize an empty zeros matrix
      dS2dvechDk = np.zeros((np.int32(nraneffs[k]*(nraneffs[k]+1)/2),1))#...

      for j in np.arange(nlevels[k]):

        # Get the indices for this level and factor.
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
                
        # Work out Z_(k,j)'V^{-1}X
        ZkjtiVX = ZtiVX[:,Ikj,:]

        # Work out the term to put into the kronecker product
        # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
        K = ZkjtiVX @ iXtiVXLt
        
        # Sum terms
        dS2dvechDk = dS2dvechDk + dupMat2D(nraneffs[k]).toarray().transpose() @ mat2vec3D(kron3D(K,K.transpose(0,2,1)))

      # Multiply by sigma^2
      dS2dvechDk = np.einsum('i,ijk->ijk',sigma2,dS2dvechDk)

      # Add to dS2
      dS2[:,DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2[:,DerivInds[k]:DerivInds[k+1]].shape)

  return(dS2)


# ============================================================================
#
# The below function calculates the derivative of Var(L\beta) with respect to
# the variance parameter vector \theta = (\sigma^2, vech(D_1),..., vech(D_r)).
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1). If we are looking at a 
#                    single random factor single random effect model 
#                    DinvIplusZtZD will only hold the diagonal elements of
#                    D(I+Z'ZD)^(-1)
# - `sigma2`: The fixed effects variance estimate.
# - `n`: The total number of observations (potentially spatially varying).
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation). If 
#          we are looking at a one random factor one random effect design 
#          the variable ZtZ only holds the diagonal elements of the matrix 
#          Z'Z.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `FisherInfoMat`: The Fisher information matrix of \theta.
#
# ============================================================================
def get_InfoMat3D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ):

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of voxels 
    v = sigma2.shape[0]

    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()

    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    FishIndsDk = np.insert(FishIndsDk,0,1)

    # Initialize FIsher Information matrix
    FisherInfoMat = np.zeros((v,tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2**2))
    
    # Add dl/dsigma2 covariance
    FisherInfoMat[:,0,0] = covdldsigma2

    
    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nraneffs)):

        # Get covariance of dldsigma and dldD      
        covdldsigma2dD = get_covdldDkdsigma23D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)[0].reshape(v,FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FisherInfoMat[:,0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigma2dD
        FisherInfoMat[:,FishIndsDk[k]:FishIndsDk[k+1],0:1] = FisherInfoMat[:,0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose((0,2,1))
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nraneffs)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk23D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)[0]

            # Add to FImat
            FisherInfoMat[np.ix_(np.arange(v), IndsDk1, IndsDk2)] = covdldDk1dDk2
            FisherInfoMat[np.ix_(np.arange(v), IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(np.arange(v), IndsDk1, IndsDk2)].transpose((0,2,1))

    # Return result
    return(FisherInfoMat)
