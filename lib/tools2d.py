import numpy as np
import scipy.sparse

# MOVE THESE
#import cvxopt
#from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack

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
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ============================================================================
#
# This function takes in a matrix and vectorizes it (i.e. transforms it
# to a vector of each of the columns of the matrix stacked on top of
# one another).
#
# ============================================================================
def mat2vec2D(matrix):
  
  #Return vectorised matrix
  return(matrix.transpose().reshape(matrix.shape[0]*matrix.shape[1],1))


# ============================================================================
#
# This function takes in a (symmetric, square) matrix and half-vectorizes
# it (i.e. transforms it to a vector of each of the columns of the matrix,
# below and including the diagonal, stacked on top of one another).
#
# ============================================================================
def mat2vech2D(matrix):
  
  # Get lower triangular indices
  rowinds, colinds = np.tril_indices(matrix.shape[0]) #Try mat.transpose()[trilu]?
  
  # They're in the wrong order so we need to order them
  # To do this we first hash them
  indhash = colinds*matrix.shape[0]+rowinds
  
  # Sort permutation
  perm=np.argsort(indhash)
  
  # Return vectorised half-matrix
  return(np.array([matrix[rowinds[perm],colinds[perm]]]).transpose())


# ============================================================================
#
# This function maps the vector of a symmetric matrix to a vector of the
# elements of the lower half of the matrix stacked column-wise.
#
# ============================================================================
def vec2vech2D(vec):
  
  # Return vech
  return(mat2vech2D(vec2mat2D(vec)))


# ============================================================================
#
# This function maps the vector created by stacking the columns of a matrix on
# top of one another to it's corresponding square matrix.
#
# ============================================================================
def vec2mat2D(vec):
  
  # Return matrix
  return(vec.reshape(np.int64(np.sqrt(vec.shape[0])),np.int64(np.sqrt(vec.shape[0]))).transpose())


# ============================================================================
#
# This function maps a vector of the elements of the lower half of a
# symmetric matrix stacked column-wise to the vector of all elements
# of the matrix, duplicates included.
#
# ============================================================================
def vech2vec2D(vech):
  
  # Return vec
  return(mat2vec2D(vech2mat2D(vech)))


# ============================================================================
#
# This function maps a vector of the elements of the lower half of a square
# symmetric matrix stacked column-wise to the original matrix.
#
# ============================================================================
def vech2mat2D(vech):
  
  # dimension of matrix
  n = np.int64((-1+np.sqrt(1+8*vech.shape[0]))/2)
  matrix = np.zeros((n,n))
  
  # Get lower triangular indices
  rowinds, colinds = np.tril_indices(matrix.shape[0])
  
  # They're in the wrong order so we need to order them
  # To do this we first hash them
  indhash = colinds*matrix.shape[0]+rowinds
  
  # Sort permutation
  perm=np.argsort(indhash)
  
  # Assign values to lower half
  matrix[rowinds[perm],colinds[perm]] = vech.reshape(vech.shape[0])
  
  # Assign values to upper half
  matrix[colinds[perm],rowinds[perm]] = vech.reshape(vech.shape[0])
  
  # Return vectorised half-matrix
  return(matrix)

# ============================================================================
#
# This function generates a duplication matrix of size n^2 by n(n+1)/2,
# which maps vech(X) to vec(X) for any symmetric n by n matrix X.
#
# ============================================================================
def dupMat2D(n):
  
  # Make vech of 1:(n(n+1)/2)
  vech = np.arange(n*(n+1)/2)
  
  # Convert to vec
  vec = vech2vec2D(vech)
  
  # Make D (sparse one hot encoded vec)
  D = scipy.sparse.csr_matrix((np.ones(n**2),(np.arange(n**2),np.int64(vec).reshape(vec.shape[0]))))
  
  return(D)


# ============================================================================
#
# This function generates the inverse duplication matrix of size n(n+1)/2
# by n^2, which maps vec(X) to vech(X) for any symmetric n by n matrix X.
#
# ============================================================================
def invDupMat2D(n):
  
  # Make vech of 1:(n(n+1)/2)
  vech = np.arange(n*(n+1)/2)
  
  # Convert to vec
  vec = np.int64(vech2vec2D(vech))
  vec = vec.reshape(vec.shape[0])
  
  # Work out frequency of each entry
  freq = 1/np.bincount(vec)
  
  # Work out duplication matrix
  D = scipy.sparse.csr_matrix((freq[vec],(vec,np.arange(n**2))))
  
  return(D)


# ============================================================================
#
# This function takes in a matrix X and returns (X+X')/2 (forces it to be
# symmetric).
#
# ============================================================================
def forceSym2D(x):
  
  # Force it to be symmetric
  return((x+x.transpose())/2)


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
def ssr2D(YtX, YtY, XtX, beta):
  
  # Return the sum of squared residuals
  return(YtY - 2*YtX @ beta + beta.transpose() @ XtX @ beta)


# ============================================================================
#
# This function gives the indices of the columns of the Z matrix which 
# correspond to factor k level j.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `k`: The grouping factor we need the columns of.*
# - `j`: The level of the grouping factor k which we are interested in.*
# - `nlevels`: A vector containing the number of levels for each factor,
#              e.g. `nlevels=[3,4]` would mean the first factor has 3
#              levels and the second factor has 4 levels.
# - `nparams`: A vector containing the number of parameters for each
#              factor, e.g. `nlevels=[2,1]` would mean the first factor
#              has 2 parameters and the second factor has 1 parameter.
#
# ---------------------------------------------------------------------------- 
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `Ikj`: The indices of the columns of Z corresponding to factor k
#          level j.
#
# *(k and j are both zero indexed)
#
# ============================================================================
def faclev_indices2D(k, j, nlevels, nparams):
  
  # Work out the starting point of the indices
  start = np.concatenate((np.array([0]), np.cumsum(nlevels*nparams)))[k] + nparams[k]*j
  
  # work out the end point of the indices
  end = start + nparams[k]
  
  return(np.arange(start, end))


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
def initBeta2D(XtX, XtY):
  
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
def initSigma22D(ete, n):

  # Return the OLS estimate of sigma
  return(1/n*ete[0,0])


# ============================================================================
#
# The function below returns an initial estimate for the Random Effects Variance matrix for the $k^{th}$ grouping factor, $D_k$. The estimator used is an adaption of the suggested estimator in Demidenko (2012) and is given by:
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
# - `lk`: The number of levels belonging to grouping factor k (l_k in the
#         above notation).
# - `ZtZ`: The Z matrix transposed and then multiplied by itself (Z'Z in the
#          above notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The OLS estimate of \sigma^2 (\sigma^2 in the above notation).
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
def initDk2D(k, lk, ZtZ, Zte, sigma2, nparams, nlevels):
  
  # Initalize D to zeros
  invSig2ZteetZminusZtZ = np.zeros((nparams[k],nparams[k]))
  
  # For each level j we need to add a term
  for j in np.arange(nlevels[k]):
    
    Ikj = faclev_indices2D(k, j, nlevels, nparams)

    # Work out Z_(k, j)'Z_(k, j)
    ZkjtZkj = ZtZ[np.ix_(Ikj,Ikj)]
    
    # Work out Z_(k,j)'e
    Zkjte = Zte[Ikj,:]
    
    if j==0:
      
      # Add first Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
      ZtZkronZtZ = np.kron(ZkjtZkj,ZkjtZkj.transpose())
      
      # Add first \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
      invSig2ZteetZminusZtZ = 1/sigma2*(Zkjte @ Zkjte.transpose()) - ZkjtZkj
      
    else:
      
      # Add next Z_(k,j)'Z_(k,j) kron Z_(k,j)'Z_(k,j)
      ZtZkronZtZ = ZtZkronZtZ + np.kron(ZkjtZkj,ZkjtZkj.transpose())
      
      # Add next \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
      invSig2ZteetZminusZtZ = invSig2ZteetZminusZtZ + 1/sigma2*(Zkjte @ Zkjte.transpose()) - ZkjtZkj
  
  # Work out the final term.
  Dkest = vec2mat2D(np.linalg.inv(ZtZkronZtZ) @ mat2vec2D(invSig2ZteetZminusZtZ)) 
  
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
def makeDnnd2D(D):
  
  # Check if we have negative eigenvalues
  if not np.all(np.linalg.eigvals(D)>0):
  
    # If we have negative eigenvalues
    eigvals,eigvecs = np.linalg.eigh(D)
    
    # Work out elementwise max of lambda and 0
    lamplus = np.diag(np.maximum(eigvals,0))
    
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
def llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D):
  
  # Work out the log likelihood
  llh = -0.5*(n*np.log(sigma2) + np.log(np.linalg.det(np.eye(ZtZ.shape[0]) + ZtZ @ D)) + (1/sigma2)*(ete - forceSym2D(Zte.transpose() @ DinvIplusZtZD @ Zte)))
  
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
def get_dldB2D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte):
  
  # Return the derivative
  return(1/sigma2*(Xte - (XtZ @ DinvIplusZtZD @ Zte)))


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
def get_dldsigma22D(n, ete, Zte, sigma2, DinvIplusZtZD):
  
  # Return the bottom expression in the above derivation
  return(-n/(2*sigma2) + 1/(2*(sigma2**2))*(ete - forceSym2D(Zte.transpose() @ DinvIplusZtZD @ Zte)))


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
def get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD):

  # Initalize the derivative to zeros
  dldDk = np.zeros((nparams[k],nparams[k]))

  # For each level j we need to add a term
  for j in np.arange(nlevels[k]):

    # Get the indices for the kth factor jth level
    Ikj = faclev_indices2D(k, j, nlevels, nparams)

    # Get (the kj^th columns of Z)^T multiplied by Z
    Z_kjtZ = ZtZ[Ikj,:]
    Z_kjte = Zte[Ikj,:]

    # Get the first term of the derivative
    Z_kjtVinve = Z_kjte - (Z_kjtZ @ DinvIplusZtZD @ Zte)
    firstterm = 1/sigma2 * forceSym2D(Z_kjtVinve @ Z_kjtVinve.transpose())
    
    # Get (the kj^th columns of Z)^T multiplied by (the kj^th columns of Z)
    Z_kjtZ_kj = ZtZ[np.ix_(Ikj,Ikj)]
    secondterm = forceSym2D(Z_kjtZ_kj) - forceSym2D(Z_kjtZ @ DinvIplusZtZD @ Z_kjtZ.transpose())
    
    if j == 0:
      
      # Start a running sum over j
      dldDk = firstterm - secondterm
      
    else:
    
      # Add these to the running sum
      dldDk = dldDk + firstterm - secondterm

  # Halve the sum (the coefficient of a half was not included in the above)
  dldDk = forceSym2D(dldDk/2)

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
def get_covdldbeta2D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2):
  
  # Return the covariance of the derivative
  return((1/sigma2)*(XtX - forceSym2D(XtZ @ DinvIplusZtZD @ XtZ.transpose())))



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
def get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False):
  
  # Sum of R_(k, j) over j
  RkSum = np.zeros(nparams[k],nparams[k])

  for j in np.arange(nlevels[k]):

    # Get the indices for the kth factor jth level
    Ikj = faclev_indices2D(k, j, nlevels, nparams)

    # Work out R_(k, j)
    Rkj = ZtZ[np.ix_(Ikj,Ikj)] - forceSym2D(ZtZ[Ikj,:] @ DinvIplusZtZD @ ZtZ[:,Ikj])

    # Add together
    RkSum = RkSum + Rkj

  # Multiply by duplication matrices and save
  if not vec:
    covdldDdldsigma2 = 1/(2*sigma2) * invDupMatdict[k] @ mat2vec2D(RkSum)
  else:
    covdldDdldsigma2 = 1/(2*sigma2) * mat2vec2D(RkSum)  

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
#          the update vector for vec (i.e. duplicates included), otherwise it
#          gives the update vector for vech.
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
def get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False):
  
  # Sum of R_(k1, k2, i, j) kron R_(k1, k2, i, j) over i and j 
  for i in np.arange(nlevels[k1]):

    for j in np.arange(nlevels[k2]):
      
      # Get the indices for the k1th factor jth level
      Ik1i = faclev_indices2D(k1, i, nlevels, nparams)
      Ik2j = faclev_indices2D(k2, j, nlevels, nparams)
      
      # Work out R_(k1, k2, i, j)
      Rk1k2ij = ZtZ[np.ix_(Ik1i,Ik2j)] - (ZtZ[Ik1i,:] @ DinvIplusZtZD @ ZtZ[:,Ik2j])
      
      # Work out Rk1k2ij kron Rk1k2ij
      RkRt = np.kron(Rk1k2ij,Rk1k2ij)
      
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
# The below function applies a mapping to a vector of parameters. (Used in PLS
# - equivalent of what is described in Bates 2015)
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `theta`: the vector of theta parameters.
# - `theta_inds`: A vector specifying how many times each theta parameter 
#                 should be repeated. For example, if theta=[0.1,0.8,0.3] 
#                 and theta_inds=[1,1,1,2,3,3], then the values to be mapped 
#                 into the sparse matrix would be [0.1,0.1,0.1,0.8,0.3,0.3].
# - `r_inds`: The row indices of the elements mapped into the sparse matrix.
# - `c_inds`: The column indices of the elements mapped into the sparse matrix.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `Lambda`: The sparse matrix containing the elements oftheta in the correct
#             indices.
#
# ============================================================================
def mapping2D(theta, theta_inds, r_inds, c_inds):

    return(spmatrix(theta[theta_inds.astype(np.int64)].tolist(), r_inds.astype(np.int64), c_inds.astype(np.int64)))
    

# ============================================================================
#
# This function takes in a square matrix M and outputs P and L from it's 
# sparse cholesky decomposition of the form PAP'=LL'.
#
# Note: P is given as a permutation vector rather than a matrix. Also 
# cholmod.options['supernodal'] must be set to 2.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `M`: The matrix to be sparse cholesky decomposed as an spmatrix from the 
#        cvxopt package.
# - `perm`: Input permutation (*optional*, one will be calculated if not)
# - `retF`: Return the factorisation object or not
# - `retP`: Return the permutation or not
# - `retL`: Return the lower cholesky or not
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `F`: A factorization object.
#
# ============================================================================
def sparse_chol2D(M, perm=None, retF=False, retP=True, retL=True):

    # Quick check that M is square
    if M.size[0]!=M.size[1]:
        raise Exception('M must be square.')

    if not perm is None:
        # Make an expression for the factorisation
        F=cholmod.symbolic(M,p=perm)
    else:
        # Make an expression for the factorisation
        F=cholmod.symbolic(M)

    # Calculate the factorisation
    cholmod.numeric(M, F)

    # Empty factorisation object
    factorisation = {}

    if (retF and retL) or (retF and retP):

        # Calculate the factorisation again (buggy if returning L for
        # some reason)
        F2=cholmod.symbolic(M,p=perm)
        cholmod.numeric(M, F2)

        # If we want to return the F object, add it to the dictionary
        factorisation['F']=F2
        
    else:
      
      factorisation['F']=F

    if retP:

        # Set p to [0,...,n-1]
        P = cvxopt.matrix(range(M.size[0]), (M.size[0],1), tc='d')

        # Solve and replace p with the true permutation used
        cholmod.solve(F, P, sys=7)

        # Convert p into an integer array; more useful that way
        P=cvxopt.matrix(np.array(P).astype(np.int64),tc='i')

        # If we want to return the permutation, add it to the dictionary
        factorisation['P']=P

    if retL:

        # Get the sparse cholesky factor
        L=cholmod.getfactor(F)
        
        # If we want to return the factor, add it to the dictionary
        factorisation['L']=L

    # Return P and L
    return(factorisation)


# ============================================================================
# This function takes in a vector of parameters, theta, and returns indices 
# which maps them the to lower triangular block diagonal matrix, lambda.
#
# ----------------------------------------------------------------------------
#
# The following inputs are required for this function:
#
# ----------------------------------------------------------------------------
#
# - `nlevels`: a vector of the number of levels for each grouping factor. 
#              e.g. nlevels=[10,2] means there are 10 levels for factor 1 and 
#              2 levels for factor 2.
# - `nparams`: a vector of the number of variables for each grouping factor. 
#              e.g. nparams=[3,4] means there are 3 variables for factor 1 and
#              4 variables for factor 2.
#
# All arrays must be np arrays.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - `theta_repeated_inds`: This is a vector that tells us how to repeat the 
#                          values in the theta vector. 
# - `row_indices`: This is the row indices we enter the theta values into.
# - `column_indices`: This is the column indices we enter the theta values 
#                     into.
#
# Example: theta_repeated_inds = [1,1,2], row_inds = [2,3,3], col_inds = [3, 2, 3]
#          This means we enter the first value of theta into elements [2,3] of 
#          [3,2] of Lambda and the second element of theta into element [3,3]
#          of Lambda.
#
# ============================================================================
def get_mapping2D(nlevels, nparams):

    # Work out how many factors there are
    n_f = len(nlevels)

    # Quick check that nlevels and nparams are the same length
    if len(nlevels)!=len(nparams):
        raise Exception('The number of parameters and number of levels should be recorded for every grouping factor.')

    # Work out how many lambda components needed for each factor
    n_lamcomps = (np.multiply(nparams,(nparams+1))/2).astype(np.int64)

    # Block index is the index of the next un-indexed diagonal element
    # of Lambda
    block_index = 0

    # Row indices and column indices of theta
    row_indices = np.array([])
    col_indices = np.array([])

    # This will have the values of theta repeated several times, once
    # for each time each value of theta appears in lambda
    theta_repeated_inds = np.array([])
    
    # Loop through factors generating the indices to map theta to.
    for i in range(0,n_f):

        # Work out the indices of a lower triangular matrix
        # of size #variables(factor) by #variables(factor)
        row_inds_tri, col_inds_tri = np.tril_indices(nparams[i])

        # Work out theta for this block
        theta_current_inds = np.arange(np.sum(n_lamcomps[0:i]),np.sum(n_lamcomps[0:(i+1)]))

        # Work out the repeated theta
        theta_repeated_inds = np.hstack((theta_repeated_inds, np.tile(theta_current_inds, nlevels[i])))

        # For each level of the factor we must repeat the lower
        # triangular matrix
        for j in range(0,nlevels[i]):

            # Append the row/column indices to the running list
            row_indices = np.hstack((row_indices, (row_inds_tri+block_index)))
            col_indices = np.hstack((col_indices, (col_inds_tri+block_index)))

            # Move onto the next block
            block_index = block_index + nparams[i]

    # Create lambda as a sparse matrix
    #lambda_theta = spmatrix(theta_repeated.tolist(), row_indices.astype(np.int64), col_indices.astype(np.int64))

    # Return lambda
    return(theta_repeated_inds, row_indices, col_indices)