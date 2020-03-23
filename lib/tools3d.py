import numpy as np
import scipy.sparse
from lib.tools2d import faclev_indices2D, permOfIkKkI2D

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
#     - PLS: https://colab.research.google.com/drive/1add6pX26d32WxfMUTXNz4wixYR1nOGi0
#     - FS: https://colab.research.google.com/drive/12CzYZjpuLbENSFgRxLi9WZfF5oSwiy-e
#     - GS: https://colab.research.google.com/drive/1sjfyDF_EhSZY60ziXoKGh4lfb737LFPD
# - Tom Maullin (12/11/2019)
#   This file contains "3D" versions of all functions given in `2dtools.py`. 
#   By 3D, I mean, where `2dtools.py` would take a matrix as input and do 
#   something to it, `3dtools.py` will take as input a 3D array (stack of 
#   matrices) and perform the operation to all matrices in the last 2 
#   dimensions. (Note: The documentation below is identical to 2dtools so 
#   beware this distinction!).
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
    return(np.einsum("ijk,ilm->ijlkm",A,B).reshape(i1,j*l,k*m))
  elif i1==1 or i2==1:
    return(np.einsum("ijk,nlm->injlkm",A,B).reshape(i1*i2,j*l,k*m))
  else:
    raise ValueError('Incompatible dimensions in kron3D.')


# ============================================================================
#
# This function takes in a matrix and vectorizes it (i.e. transforms it
# to a vector of each of the columns of the matrix stacked on top of
# one another).
#
# ============================================================================
def mat2vec3D(matrix):
  
  #Return vectorised matrix
  return(matrix.transpose(0,2,1).reshape(matrix.shape[0],matrix.shape[1]*matrix.shape[2],1))


# ============================================================================
#
# This function takes in a (symmetric, square) matrix and half-vectorizes
# it (i.e. transforms it to a vector of each of the columns of the matrix,
# below and including the diagonal, stacked on top of one another).
#
# ============================================================================
def mat2vech3D(matrix):
  
  # Number of voxels, nv
  nv = matrix.shape[0]
  
  # Get lower triangular indices
  rowinds, colinds = np.tril_indices(matrix.shape[1]) 
  
  # Number of covariance parameters, nc
  nc = len(rowinds)
  
  # They're in the wrong order so we need to order them
  # To do this we first hash them
  indhash = colinds*matrix.shape[1]+rowinds
  
  # Sort permutation
  perm=np.argsort(indhash)
  
  # Return vectorised half-matrix
  return(matrix[:,rowinds[perm],colinds[perm]].reshape((nv,nc,1)))


# ============================================================================
#
# This function takes in a stack of vectors and returns the corresponding
# matrix forms treating the elements of the vectors as the elements of lower
# halves of those matrices.
#
# ============================================================================
def vech2mat3D(vech):
  
  # Number of voxels
  nv = vech.shape[0]
  
  # dimension of matrix
  n = np.int64((-1+np.sqrt(1+8*vech.shape[1]))/2)
  matrix = np.zeros((nv,n,n))
  
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
  return(vec.reshape(vec.shape[0], np.int64(np.sqrt(vec.shape[1])),np.int64(np.sqrt(vec.shape[1]))).transpose(0,2,1))

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
  return((x+x.transpose((0,2,1)))/2)


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
  return(YtY - 2*YtX @ beta + beta.transpose((0,2,1)) @ XtX @ beta)


# ============================================================================
#
# This function takes in a dictionary, `Ddict`, in which entry `k` is a stack 
# of the kth diagonal block for every voxel.
#
# ============================================================================
def getDfromDict3D(Ddict, nparams, nlevels):
  
  # Get number of voxels
  nv = Ddict[0].shape[0]
  
  # Work out indices (there is one block of D per level)
  inds = np.zeros(np.sum(nlevels)+1)
  counter = 0
  for k in np.arange(len(nparams)):
    for j in np.arange(nlevels[k]):
      inds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nparams)))[k] + nparams[k]*j
      counter = counter + 1
      
  
  # Last index will be missing so add it
  inds[len(inds)-1]=inds[len(inds)-2]+nparams[-1]
  
  # Make sure indices are ints
  inds = np.int64(inds)
  
  # Initial D
  D = np.zeros((nv,np.sum(nparams*nlevels),np.sum(nparams*nlevels)))

  counter = 0
  for k in np.arange(len(nparams)):
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
# - `n`: The total number of observations (n in the above notation).
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

  # Return the OLS estimate of sigma
  return(1/n*ete[:,0,0])


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
# - `ZtZ`: The Z matrix transposed and then multiplied by itself (Z'Z in the
#          above notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The OLS estimate of \sigma^2 (\sigma^2$ in the above notation).
# - `invDupMatdict`: A dictionary of inverse duplication matrices such that 
#                   `invDupMatdict[k]` = DupMat_k^+.
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
def initDk3D(k, ZtZ, Zte, sigma2, nlevels, nparams, invDupMatdict):
  
  # Small check on sigma2
  if len(sigma2.shape) > 1:

    sigma2 = sigma2.reshape(sigma2.shape[0])

  # Initalize D to zeros
  invSig2ZteetZminusZtZ = np.zeros((Zte.shape[0],nparams[k],nparams[k]))

  # First we work out the derivative we require.
  for j in np.arange(nlevels[k]):
    
    Ikj = faclev_indices2D(k, j, nlevels, nparams)

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

  # Second we need to work out the double sum of Z_(k,j)'Z_(k,j)
  for j in np.arange(nlevels[k]):

    for i in np.arange(nlevels[k]):
      
      Iki = faclev_indices2D(k, i, nlevels, nparams)
      Ikj = faclev_indices2D(k, j, nlevels, nparams)

      # Work out Z_(k, j)'Z_(k, j)
      ZkitZkj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Iki,Ikj)]
      
      if j==0 and i==0:
        
        # Add first Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
        ZtZkronZtZ = kron3D(ZkitZkj,ZkitZkj.transpose(0,2,1))
     
      else:
        
        # Add next Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
        ZtZkronZtZ = ZtZkronZtZ + kron3D(ZkitZkj,ZkitZkj.transpose(0,2,1))

  # Work out information matrix
  infoMat = invDupMatdict[k].toarray() @ ZtZkronZtZ @ invDupMatdict[k].toarray().transpose()

  # Work out the final term.
  Dkest = vech2mat3D(np.linalg.inv(infoMat) @ mat2vech3D(invSig2ZteetZminusZtZ)) 
  
  
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
# - `D`: A square symmetric matrix.
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
# - `n`: The total number of observations.
# - `ZtZ`: The Z matrix transposed and then multiplied by Z (Z'Z in the above
#          notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `ete`: The OLS residuals transposed and then multiplied by themselves
#          (e'e=(Y-X\beta)'(Y-X\beta) in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
# - `D`: The random effects variance-covariance matrix (D in the above
#        notation)
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
def llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D,reml=False, XtX=0, XtZ=0, ZtX=0):
  
  if hasattr(n, "ndim"):

    if np.prod(n.shape) > 1:

      n = n.reshape(sigma2.shape)
  
  # Work out -1/2(nln(sigma^2) + ln|I+Z'ZD|)
  if reml==False:
    firstterm = -0.5*(n*np.log(sigma2)).reshape(ete.shape[0]) - 0.5*np.log(np.linalg.det(np.eye(ZtZ.shape[1]) + D @ ZtZ)).reshape(ete.shape[0])
  else:
    p = XtX.shape[1]
    firstterm = -0.5*((n-p)*np.log(sigma2)).reshape(ete.shape[0]) -0.5*np.log(np.linalg.det(np.eye(ZtZ.shape[1]) + D @ ZtZ)).reshape(ete.shape[0])


  # Work out sigma^(-2)*(e'e - e'ZD(I+Z'ZD)^(-1)Z'e)
  secondterm = -0.5*np.einsum('i,ijk->ijk',(1/sigma2).reshape(ete.shape[0]),(ete - forceSym3D(Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte))).reshape(ete.shape[0])
  
  # Work out the log likelihood
  llh = (firstterm + secondterm).reshape(ete.shape[0])

  if reml:
    llh = llh - 0.5*np.log(np.linalg.det(XtX - XtZ @ DinvIplusZtZD @ ZtX))
  
  # Return result
  return(llh)


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
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
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
def get_dldB3D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte):

  # Work out the derivative (Note: we leave everything as 3D for ease of future computation)
  deriv = np.einsum('i,ijk->ijk',1/sigma2, (Xte - (XtZ @ DinvIplusZtZD @ Zte)))
                    
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
# - `n`: The number of observations.
# - `ete`: The OLS residuals transposed and then multiplied by themselves
#         (e'e=(Y-X\beta)'(Y-X\beta) in the above notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
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
def get_dldsigma23D(n, ete, Zte, sigma2, DinvIplusZtZD):
  
  # Make sure n is correct shape
  if hasattr(n, "ndim"):

    if np.prod(n.shape) > 1:

      n = n.reshape(sigma2.shape)

  # Get e'(I+ZDZ')^(-1)e=e'e-e'ZD(I+Z'ZD)^(-1)Z'e
  etinvIplusZtDZe = ete - forceSym3D(Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)
  
  # Get the derivative
  deriv = -n/(2*sigma2) + np.einsum('i,ijk->ijk',1/(2*(sigma2**2)), etinvIplusZtDZe).reshape(sigma2.shape[0])
  
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
# - `nparams`: A vector containing the number of parameters for each factor,
#              e.g. `nlevels=[2,1]` would mean the first factor has 2
#              parameters and the second factor has 1 parameter.
# - `ZtZ`: The Z matrix transposed and then multiplied by itself (Z'Z in the
#          above notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `dldDk`: The derivative of l with respect to D_k.
#
# ============================================================================
def get_dldDk3D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD,reml=False, ZtX=0, XtX=0):

  # Number of voxels
  nv = Zte.shape[0]
  
  # Initalize the derivative to zeros
  dldDk = np.zeros((nv, nparams[k],nparams[k]))
  
  # For each level j we need to add a term
  for j in np.arange(nlevels[k]):

    # Get the indices for the kth factor jth level
    Ikj = faclev_indices2D(k, j, nlevels, nparams)
    
    # Get (the kj^th columns of Z)^T multiplied by Z
    Z_kjtZ = ZtZ[:,Ikj,:]
    Z_kjte = Zte[:,Ikj,:]
    
    # Get the first term of the derivative
    Z_kjtVinve = Z_kjte - (Z_kjtZ @ DinvIplusZtZD @ Zte)
    firstterm = np.einsum('i,ijk->ijk',1/sigma2,forceSym3D(Z_kjtVinve @ Z_kjtVinve.transpose((0,2,1))))
    
    # Get (the kj^th columns of Z)^T multiplied by (the kj^th columns of Z)
    Z_kjtZ_kj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)]
    secondterm = forceSym3D(Z_kjtZ_kj) - forceSym3D(Z_kjtZ @ DinvIplusZtZD @ Z_kjtZ.transpose((0,2,1)))
    
    if j == 0:
      
      # Start a running sum over j
      dldDk = firstterm - secondterm
      
    else:
    
      # Add these to the running sum
      dldDk = dldDk + firstterm - secondterm
    
  if reml==True:

    invXtinvVX = np.linalg.inv(XtX - ZtX.transpose((0,2,1)) @ DinvIplusZtZD @ ZtX)

    # For each level j we need to add a term
    for j in np.arange(nlevels[k]):

      # Get the indices for the kth factor jth level
      Ikj = faclev_indices2D(k, j, nlevels, nparams)

      Z_kjtZ = ZtZ[:,Ikj,:]
      Z_kjtX = ZtX[:,Ikj,:]

      Z_kjtinvVX = Z_kjtX - Z_kjtZ @ DinvIplusZtZD @ ZtX

      dldDk = dldDk + 0.5*Z_kjtinvVX @ invXtinvVX @ Z_kjtinvVX.transpose((0,2,1))

  # Halve the sum (the coefficient of a half was not included in the above)
  dldDk = forceSym3D(dldDk/2)

  # Store it in the dictionary
  return(dldDk)


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
# - `XtZ`: X transpose multiplied by Z.
# - `XtX`: X transpose multiplied by X.
# - `ZtZ`: Z transpose multiplied by Z.
# - `DinvIplusZtZD`: D(I+Z'ZD)^(-1) in the above notation.
# - `sigma2`: The fixed effects variance (\sigma^2 in the above notation).
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
def get_covdldbeta3D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2):
  
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
# - `nparams`: A vector containing the number of parameters for each factor, 
#              e.g. `nlevels=[2,1]` would mean the first factor has 2 
#              parameters and the second factor has 1 parameter.
# - `ZtZ`: Z transpose multiplied by Z.
# - `DinvIplusZtZD`: D(I+Z'ZD)^(-1) in the above notation.
# - `invDupMatdict`: A dictionary of inverse duplication matrices such that 
#                   `invDupMatdict[k]` = DupMat_k^+.
# - `vec`: This is a boolean value which by default is false. If True it gives
#          the update vector for vec (i.e. duplicates included), otherwise it
#          gives the update vector for vech.
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
#
# ============================================================================
def get_covdldDkdsigma23D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False):
  
  # Number of voxels
  nv = DinvIplusZtZD.shape[0]
  
  # Sum of R_(k, j) over j
  RkSum = np.zeros((nv,nparams[k],nparams[k]))

  for j in np.arange(nlevels[k]):

    # Get the indices for the kth factor jth level
    Ikj = faclev_indices2D(k, j, nlevels, nparams)

    # Work out R_(k, j)
    Rkj = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ikj,Ikj)] - forceSym3D(ZtZ[:,Ikj,:] @ DinvIplusZtZD @ ZtZ[:,:,Ikj])
    
    # Add together
    RkSum = RkSum + Rkj

  # Multiply by duplication matrices and 
  if not vec:
    covdldDdldsigma2 = np.einsum('i,ijk->ijk', 1/(2*sigma2), invDupMatdict[k] @ mat2vec3D(RkSum))
  else:
    covdldDdldsigma2 = np.einsum('i,ijk->ijk', 1/(2*sigma2), mat2vec3D(RkSum))
  
  return(covdldDdldsigma2)


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
# - `nparams`: A vector containing the number of parameters for each factor,
#              e.g. `nlevels=[2,1]` would mean the first factor has 2 
#              parameters and the second factor has 1 parameter.
# - `ZtZ`: Z transpose multiplied by Z.
# - `DinvIplusZtZD`: D(I+Z'ZD)^(-1) in the above notation.
# - `invDupMatdict`: A dictionary of inverse duplication matrices such that 
#                    `invDupMatdict[k]` = DupMat_k^+
# - `vec`: This is a boolean value which by default is false. If True it gives
#          the update matrix for vec (i.e. duplicates included), otherwise it
#          gives true covariance for vech.
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
#
# ============================================================================
def get_covdldDk1Dk23D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False):
  
  # Sum of R_(k1, k2, i, j) kron R_(k1, k2, i, j) over i and j 
  for i in np.arange(nlevels[k1]):

    for j in np.arange(nlevels[k2]):
      
      # Get the indices for the k1th factor jth level
      Ik1i = faclev_indices2D(k1, i, nlevels, nparams)
      Ik2j = faclev_indices2D(k2, j, nlevels, nparams)
      
      # Work out R_(k1, k2, i, j)
      Rk1k2ij = ZtZ[np.ix_(np.arange(ZtZ.shape[0]),Ik1i,Ik2j)] - (ZtZ[:,Ik1i,:] @ DinvIplusZtZD @ ZtZ[:,:,Ik2j])
      
      # Work out Rk1k2ij kron Rk1k2ij
      RkRt = kron3D(Rk1k2ij,Rk1k2ij)
      
      # Add together
      if (i == 0) and (j == 0):
      
        RkRtSum = RkRt
      
      else:
        
        RkRtSum = RkRtSum + RkRt
    
  # Multiply by duplication matrices and save
  if not vec:
    covdldDk1dldk2 = 1/2 * invDupMatdict[k1] @ RkRtSum @ invDupMatdict[k2].transpose()
  else:
    covdldDk1dldk2 = 1/2 * RkRtSum


  # Return the result
  return(covdldDk1dldk2)



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
#  - A: A 3D matrix of dimension (v by m1 by m2).
#  - B: A 3D matrix of dimension (v by m1' by m2).
#  - pA: The size of the block partitions of A, e.g. if Av_{i,j} is of 
#        dimension (n1 by n2) then pA=[n1, n2].
#  - pB: The size of the block partitions of B, e.g. if Bv_{i,j} is of 
#        dimension (n1' by n2) the pB=[n1', n2].
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
  
  # Number of voxels
  v = A.shape[0]

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
  S = A.transpose((0,2,1)).reshape((v,mA,nA)).transpose((0,2,1)) @ B.transpose((0,2,1)).reshape((v,mB,nB))

  # Return result
  return(S)


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
#  - `p`: The size of the block partitions of A and B, e.g. if A_{i,j} and 
#         B_{i,j} are of dimension (n1 by n2) then pA=[n1, n2].
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
def sumAijKronBij3D(A, B, p, perm=None):

  # Check dim A and B and pA and pB all same
  n1 = p[0]
  n2 = p[1]

  # Number of voxels
  v = A.shape[0]

  # This matrix only needs be calculated once
  if perm is None:
    perm = permOfIkKkI2D(n2,n1,n2,n1) 

  # Convert to vecb format
  atilde = mat2vecb3D(A,p)
  btilde = mat2vecb3D(B,p)

  # Multiply and convert to vector
  vecba = mat2vec3D(btilde.transpose((0,2,1)) @ atilde)

  # Permute
  S_noreshape = vecba[:,perm,:] 

  # Reshape to correct shape
  S = S_noreshape.reshape(v,n2**2,n1**2).transpose((0,2,1))

  return(S,perm)
