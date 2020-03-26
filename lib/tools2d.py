import numpy as np
import scipy.sparse
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack

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
  rowinds, colinds = np.tril_indices(matrix.shape[0])
  
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
# The below function inverts a block diagonal matrix with square diagonal 
# blocks by inverting each block individually.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - `matrix`: The (scipy sparse) block diagonal matrix to be inverted.
#  - `blockSize`: The size of the blocks on the diagonal. I.e. if block-size 
#                 equals 5 then all blocks on the diagonal are 5 by 5 in size.
#
# ----------------------------------------------------------------------------
#
# This function gives as outputs:
#
# ----------------------------------------------------------------------------
#
#  - `invMatrix`: The inverse of `matrix`, in (scipy) sparse format.
#
# ----------------------------------------------------------------------------
#
# Developers note: Whilst this function, in principle, makes a lot of sense, 
# in practice it does not outperform numpy and, unless Z'Z is so ridiculously
# large it cannot be read into memory, I do not think there will ever be 
# reason to use it. I have left it here just incase it can be useful in future
# applications. (It does outperform scipy though!)
#
# ============================================================================
def blockInverse2D(matrix, blockSize):

  # Work out the number of blocks
  numBlocks = matrix.shape[0]//blockSize

  # For each level, invert the corresponding block on the diagonal
  for i in range(numBlocks):
    
    # The block is nparams by nparams
    blockInds = np.ix_(np.arange(i*blockSize,(i+1)*blockSize),np.arange(i*blockSize,(i+1)*blockSize))
    
    # Get the block
    block = matrix[blockInds].toarray()
    
    # Replace it with it's inverse
    if i==0:

      invMatrix=np.linalg.inv(block)

    else:

      invMatrix=scipy.sparse.block_diag((invMatrix, np.linalg.inv(block)))
    
  return(invMatrix)


# ============================================================================
# 
# The below function inverts symmetric matrices with the same structure as Z'Z 
# efficiently by recursively using the schur complement and noting that the 
# only submatrices that need to be inverted are block diagonal.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - `M`: The symmetric matrix, with structure similar to Z'Z, to be inverted.
#  - `nparams`: A vector containing the number of parameters for each
#               factor, e.g. `nlevels=[2,1]` would mean the first factor
#               has 2 parameters and the second factor has 1 parameter.
#  - `nlevels`: A vector containing the number of levels for each factor,
#               e.g. `nlevels=[3,4]` would mean the first factor has 3
#               levels and the second factor has 4 levels.
#
# ----------------------------------------------------------------------------
#
# This function gives as outputs:
#
# ----------------------------------------------------------------------------
#
#  - `Minv`: The inverse of `M`, in (scipy) sparse format.
#
# ----------------------------------------------------------------------------
#
# Developers note: Again, whilst this function, in principle, makes a lot of 
# sense, in practice it was slow and is left here only for future reference.
#
# ============================================================================
def recursiveInverse2D(M, nparams, nlevels):
  
  # Check if we have a matrix we can partition into more than 1 block
  if len(nparams) > 1:
  
    # Work out qc (current q)
    qc = nparams[-1]*nlevels[-1]
    # Make q
    q = M.shape[0]

    # Get A, B and C where M=[[A,B],[B',C]]
    # A
    A_inds = np.ix_(np.arange(0,(q-qc)),np.arange(0,(q-qc)))
    A = M[A_inds]

    # B
    B_inds = np.ix_(np.arange(0,(q-qc)),np.arange((q-qc),q))
    B = M[B_inds].toarray() # B is dense

    # C
    C_inds = np.ix_(np.arange((q-qc),q),np.arange((q-qc),q))
    C = M[C_inds].toarray() # C is small and now only involved in dense mutliplys

    # Recursive inverse A
    if nparams[:-1].shape[0] > 1:

      Ainv = scipy.sparse.csr_matrix(recursiveInverse2D(A, nparams[:-1], nlevels[:-1])).toarray()

    else:

      #Ainv = blockInverse(A, nparams[0], nlevels[0]) - much slower
      Ainv = scipy.sparse.csr_matrix(scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(A))).toarray()

    # Schur complement
    S = C - B.transpose() @ Ainv @ B
    Sinv = np.linalg.inv(S)

    # Top Left Hand Side of inverse
    TLHS = Ainv + Ainv @ B @ Sinv @ B.transpose() @ Ainv


    # Top Right Hand Side of inverse
    TRHS = - Ainv @ B @ Sinv


    # Bottom Right Hand Side of inverse
    BRHS = Sinv

    # Join together
    top = np.hstack((TLHS,TRHS))
    bottom = np.hstack((TRHS.transpose(), BRHS))

    # Make Minv
    Minv = np.vstack((top, bottom))
    
  else:
    
    # If we have only one block; invert it
    Minv = scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(M)).toarray() 
  
  return(Minv)


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
# This function returns the commutation matrix of dimensions a times b.
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
#  - a: A postive integer.
#  - b: A postive integer.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - K: The commutation matrix (in sparse format) which maps vec(A) to vec(A')
#       for an arbitrary matrix A of dimensions (a,b), i.e. K is the unique
#       matrix which satisfies, for all A;
#
#                 Kvec(A) = vec(A')
# ============================================================================
def comMat2D(a, b):

  # Get row indices
  row  = np.arange(a*b)

  # Get column indices
  col  = row.reshape((a, b), order='F').ravel()

  # Ones to put in the matrix
  data = np.ones(a*b, dtype=np.int8)

  # Sparse it
  K = scipy.sparse.csr_matrix((data, (row, col)), shape=(a*b, a*b))
  
  # Return K
  return(K)


# ============================================================================
# 
# The below function calculates the permutation vector corresponding to 
# multiplying a matrix, A, by the matrix I_{k1} kron K_{n1,k2} kron I_{n2},
# where I_{j} is the (jxj) identity matrix and K_{i,j} is the (i,j) 
# commutation matrix (see comMat2D).
#
# ----------------------------------------------------------------------------
#
# The below function takes as inputs:
#
# ----------------------------------------------------------------------------
#
#  - k1: A positive integer.
#  - k2: A positive integer.
#  - n1: A positive integer.
#  - n2: A positive integer.
#
# ----------------------------------------------------------------------------
#
# And returns the permutation vector p such that for any matrix A of 
# appropriate dimensions.
#
# (I_{k1} kron K_{n1,k2} kron I_{n2}) A = A_p
#
# Where A_p is the matrix A with p applied to it's rows and K_{n1,k2} is the 
# (n1,k2), commutation matrix.
#
# ============================================================================
def permOfIkKkI2D(k1,k2,n1,n2):

  # First we need the permutation represented by matrix K in vector format
  permP = np.arange(n1*k2).reshape((n1, k2), order='F').ravel()

  # Now we work out the permutation obtained by the first kronecker product (i.e. I kron K)
  permKron1 = np.repeat(np.arange(k1),n1*k2)*n1*k2+np.tile(permP,k1)

  # Now we work out the final permutation by appplying the second kronecker product (i.e. I kron K kron I)
  p = np.repeat(permKron1,n2)*n2+np.tile(np.arange(n2),n1*k1*k2)

  # Return the permutation
  return(p)


# ============================================================================
# 
# This function converts a matrix partitioned into blocks into a matrix 
# consisting of each block stacked on top of one another. I.e. it maps matrix
# A to matrix A_s like so:
#
#                                                      |   A_{1,1}   |
#                                                      |   A_{1,2}   |
#     | A_{1,1}    A_{1,2}  ...  A_{1,l_2}  |          |    ...      |
#     | A_{2,1}    A_{2,2}  ...  A_{2,l_2}  |          |  A_{1,l_2}  |
# A = |  ...         ...    ...      ...    | -> A_s = |   A_{2,1}   |
#     | A_{l_1,1} A_{l_1,2} ... A_{l_1,l_2} |          |    ...      |
#                                                      |    ...      |
#                                                      | A_{l_1,l_2} |
#
# ----------------------------------------------------------------------------
#
# This function takes as inputs:
# 
# ----------------------------------------------------------------------------
#
#  - A: A 2D matrix of dimension (m1 by m2).
#  - pA: The size of the block partitions of A, e.g. if A_{i,j} is of dimension
#        (n1 by n2) then pA=[n1, n2].
# 
# ----------------------------------------------------------------------------
#
# And returns as output:
#
# ----------------------------------------------------------------------------
#
#  - As: The matrix A reshaped to have all blocks A_{i,j} on top of one 
#        another. I.e. the above mapping has been performed.
#
# ============================================================================
def block2stacked2D(A, pA):

  # Work out shape of A
  m1 = A.shape[0]
  m2 = A.shape[1]

  # Work out shape of As
  n1 = pA[0]
  n2 = pA[1]
  
  # Change A to stacked form
  As = A.reshape((m1//n1,n1,m2//n2,n2)).transpose(0,2,1,3).reshape(m1*m2//n2,n2)

  return(As)


# ============================================================================
# 
# This function converts a matrix partitioned into blocks into a matrix 
# consisting of each block converted to a vector and stacked on top of one
# another. I.e. it maps matrix A to matrix vecb(A) (``vec-block" of A) like so:
#
#                                                          |   vec'(A_{1,1})   |
#                                                          |   vec'(A_{1,2})   |
#     | A_{1,1}    A_{1,2}  ...  A_{1,l_2}  |              |        ...        |
#     | A_{2,1}    A_{2,2}  ...  A_{2,l_2}  |              |  vec'(A_{1,l_2})  |
# A = |  ...         ...    ...      ...    | -> vecb(A) = |   vec'(A_{2,1})   |
#     | A_{l_1,1} A_{l_1,2} ... A_{l_1,l_2} |              |        ...        |
#                                                          |        ...        |
#                                                          | vec'(A_{l_1,l_2}) |
#
# ----------------------------------------------------------------------------
#
# The below function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - mat: An abritary matrix whose dimensions are multiples of p[0] and p[1]
#         respectively.
#  - p: The size of the blocks we are partitioning mat into.
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
#          where A_{i,j} has dimensions (p[0] by p[1]) for all i and j.
#
# ============================================================================
def mat2vecb2D(mat,p):

  # Change to stacked block format, if necessary
  if p[1]!=mat.shape[1]:
    mat = block2stacked2D(mat,p)

  # Get height of block.
  n = p[0]
  
  # Work out shape of matrix.
  m = mat.shape[0]
  k = mat.shape[1]

  # Convert to stacked vector format
  vecb = mat.reshape(m//n, n, k).transpose((1, 0, 2)).reshape(n, m*k//n).transpose().reshape(m//n,n*k)

  #Return vecb
  return(vecb)


# ============================================================================
#
# The below function computes, given two matrices A and B the below sum:
#
#                 S = Sum_i Sum_j (A_{i,j}B_{i,j}')
# 
# where the matrices A and B are block partitioned like so:
#
#     |   A_{1,1}  ...  A_{1,l2}  |      |   B_{1,1}  ...  B_{1,l2}  | 
# A = |    ...     ...     ...    |  B = |    ...     ...     ...    | 
#     |  A_{l1,1}  ...  A_{l1,l2} |      |  B_{l1,1}  ...  B_{l1,l2} | 
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - A: A 2D matrix of dimension (m1 by m2).
#  - B: A 2D matrix of dimension (m1' by m2).
#  - pA: The size of the block partitions of A, e.g. if A_{i,j} is of 
#        dimension (n1 by n2) then pA=[n1, n2].
#  - pB: The size of the block partitions of B, e.g. if B_{i,j} is of 
#        dimension (n1' by n2) the pB=[n1', n2].
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
#  - S: The sum of the partitions of A multiplied by the transpose of the 
#       partitions of B.
# 
# ----------------------------------------------------------------------------
#
# Developer note: Note that the above implies that l1 must equal m1/n1=m1'/n1'
#                 and l2=m2/n2.
#
# ============================================================================
def sumAijBijt2D(A, B, pA, pB):
  
  # Work out second (the common) dimension of the reshaped A and B
  nA = pA[0]
  nB = pB[0]

  # Work out the first (the common) dimension of reshaped A and B
  mA = A.shape[0]*A.shape[1]//nA
  mB = B.shape[0]*B.shape[1]//nB

  # Check mA equals mB
  if mA != mB:
    raise Exception('Matrix dimensions incompatible.')

  # Convert both matrices to stacked block format.
  A = block2stacked2D(A,pA)
  B = block2stacked2D(B,pB)

  # Work out the sum
  S = A.transpose().reshape((mA,nA)).transpose() @ B.transpose().reshape((mB,nB))

  # Return result
  return(S)


# ============================================================================
#
# The below function computes, given two matrices A and B the below sum:
#
#                 S = Sum_i Sum_j (A_{i,j} kron B_{i,j})
# 
# where the matrices A and B are block partitioned like so:
#
#     |   A_{1,1}  ...  A_{1,l2}  |      |   B_{1,1}  ...  B_{1,l2}  | 
# A = |    ...     ...     ...    |  B = |    ...     ...     ...    | 
#     |  A_{l1,1}  ...  A_{l1,l2} |      |  B_{l1,1}  ...  B_{l1,l2} | 
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
#  - `A`: A 2D matrix of dimension (m1 by m2).
#  - `B`: A 2D matrix of dimension (m1 by m2).
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
# - `S`: The sum of the partitions of A multiplied by the transpose of the 
#        partitions of B; i.e. the sum given above.
# - `perm`: The permutation (same as input) used for calculation (useful for 
#           later computation).
#
# ============================================================================
def sumAijKronBij2D(A, B, p, perm=None):

  # Check dim A and B and pA and pB all same
  n1 = p[0]
  n2 = p[1]

  # This matrix only needs be calculated once
  if perm is None:
    perm = permOfIkKkI2D(n2,n1,n2,n1) 

  # Convert to vecb format
  atilde = mat2vecb2D(A,p)
  btilde = mat2vecb2D(B,p)

  # Multiply and convert to vector
  vecba = mat2vec2D(btilde.transpose() @ atilde)

  # Permute
  S_noreshape = vecba[perm,:] 

  # Reshape to correct shape
  S = S_noreshape.reshape(n2**2,n1**2).transpose()

  return(S,perm)


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
# correspond to factor k.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `k`: The grouping factor we need the columns of.*
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
# - `Ik`: The indices of the columns of Z corresponding to factor k.
#
# *(k is zero indexed)
#
# ============================================================================
def fac_indices2D(k, nlevels, nparams):
  
  # Get indices for all factors
  allInds = np.concatenate((np.array([0]),np.cumsum(nlevels*nparams)))

  # Work out the first index
  start = allInds[k]

  # Work out the last index
  end = allInds[k+1]

  return(np.arange(start,end))

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
# - `ZtZ`: The Z matrix transposed and then multiplied by itself (Z'Z in the
#          above notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The OLS estimate of \sigma^2 (\sigma^2 in the above notation).
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
def initDk2D(k, ZtZ, Zte, sigma2, nlevels, nparams, invDupMatdict):
  
  # Initalize to zeros
  ZkjtZkj = np.zeros((nparams[k],nparams[k]))

  # First we work out the derivative we require.
  for j in np.arange(nlevels[k]):
    
    Ikj = faclev_indices2D(k, j, nlevels, nparams)

    # Work out Z_(k, j)'Z_(k, j)
    ZkjtZkj = ZkjtZkj + ZtZ[np.ix_(Ikj,Ikj)]
    
  # Work out block size
  qk = nparams[k]
  p = np.array([qk,1])

  # Get indices
  Ik = fac_indices2D(k, nlevels, nparams)

  # Work out the sum of Z_{(k,j)}'ee'Z_{(k,j)}
  ZteetZ = sumAijBijt2D(Zte[Ik,:],Zte[Ik,:],p,p)

  # Add first \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
  invSig2ZteetZminusZtZ = 1/sigma2*ZteetZ - ZkjtZkj
  
  # Second we need to work out the double sum of Z_(k,j)'Z_(k,j)
  p = np.array([nparams[k],nparams[k]])

  # Get sum of Z_{(k,j)}'Z_{(k,j)} kron Z_{(k,j)}'Z_{(k,j)}
  ZtZkronZtZ,_ = sumAijKronBij2D(ZtZ[np.ix_(Ik,Ik)], ZtZ[np.ix_(Ik,Ik)], p, perm=None)

  # Work out information matrix
  infoMat = invDupMatdict[k] @ ZtZkronZtZ @ invDupMatdict[k].transpose()

  # Work out the final term.
  Dkest = vech2mat2D(np.linalg.inv(infoMat) @ mat2vech2D(invSig2ZteetZminusZtZ)) 
  
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
  llh = -0.5*(n*np.log(sigma2) + np.prod(np.linalg.slogdet(np.eye(ZtZ.shape[0]) + ZtZ @ D)) + (1/sigma2)*(ete - forceSym2D(Zte.transpose() @ DinvIplusZtZD @ Zte)))
  
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
# - `ZtZmat` (optional): The sum over j of Z_{(k,j)}'Z_{(k,j)}. This only
#                        need be calculated once so can be stored and 
#                        re-entered for each iteration.
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
def get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD, ZtZmat=None):

  # We only need calculate this once across all iterations
  if ZtZmat is None:

    # Instantiate to zeros
    ZtZmat = np.zeros(nparams[k],nparams[k])

    for j in np.arange(nlevels[k]):

      # Get the indices for the kth factor jth level
      Ikj = faclev_indices2D(k, j, nlevels, nparams)

      # Work out R_(k, j)
      ZtZterm = ZtZ[np.ix_(Ikj,Ikj)]

      # Add together
      ZtZmat = ZtZmat + ZtZterm

  # Get the indices for the factors 
  Ik = fac_indices2D(k, nlevels, nparams)

  # Work out lk
  lk = nlevels[k]

  # Work out block size
  qk = nparams[k]
  p = np.array([qk,1])

  # Work out the second term in TT'
  secondTerm = sumAijBijt2D(ZtZ[Ik,:] @ DinvIplusZtZD, ZtZ[Ik,:], p, p)

  # Obtain RkSum=sum (TkjTkj')
  RkSum = ZtZmat - secondTerm

  # Work out T_ku*sigma
  TuSig = Zte[Ik,:] - (ZtZ[Ik,:] @ DinvIplusZtZD @ Zte)

  # Obtain Sum Tu(Tu)'
  TuuTSum = sumAijBijt2D(TuSig, TuSig, p, p)/sigma2

  # Work out dldDk
  dldDk = 0.5*(forceSym2D(TuuTSum - RkSum))

  # Store it in the dictionary
  return(dldDk,ZtZmat)


# ============================================================================
#
# Commented out below is an older version of the above code. This has been 
# left here in case it has any use for future development.
#
# ============================================================================
# def get_dldDk2D_old(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD):
#
#   # Initalize the derivative to zeros
#   dldDk = np.zeros((nparams[k],nparams[k]))
#
#   # For each level j we need to add a term
#   for j in np.arange(nlevels[k]):
#
#     # Get the indices for the kth factor jth level
#     Ikj = faclev_indices2D(k, j, nlevels, nparams)
#
#     # Get (the kj^th columns of Z)^T multiplied by Z
#     Z_kjtZ = ZtZ[Ikj,:]
#     Z_kjte = Zte[Ikj,:]
#
#     # Get the first term of the derivative
#     Z_kjtVinve = Z_kjte - (Z_kjtZ @ DinvIplusZtZD @ Zte)
#     firstterm = 1/sigma2 * forceSym2D(Z_kjtVinve @ Z_kjtVinve.transpose())
#    
#     # Get (the kj^th columns of Z)^T multiplied by (the kj^th columns of Z)
#     Z_kjtZ_kj = ZtZ[np.ix_(Ikj,Ikj)]
#     secondterm = forceSym2D(Z_kjtZ_kj) - forceSym2D(Z_kjtZ @ DinvIplusZtZD @ Z_kjtZ.transpose())
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
#   # Halve the sum (the coefficient of a half was not included in the above)
#   dldDk = forceSym2D(dldDk/2)
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
def get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False, ZtZmat=None):

  # We only need calculate this once across all iterations
  if ZtZmat is None:

    # Instantiate to zeros
    ZtZmat = np.zeros(nparams[k],nparams[k])

    for j in np.arange(nlevels[k]):

      # Get the indices for the kth factor jth level
      Ikj = faclev_indices2D(k, j, nlevels, nparams)

      # Work out R_(k, j)
      ZtZterm = ZtZ[np.ix_(Ikj,Ikj)]

      # Add together
      ZtZmat = ZtZmat + ZtZterm

  # Get the indices for the factors 
  Ik = fac_indices2D(k, nlevels, nparams)

  # Work out lk
  lk = nlevels[k]

  # Work out block size
  q = np.sum(nlevels*nparams)
  qk = nparams[k]
  p = np.array([qk,q])

  # Work out the second term
  secondTerm = sumAijBijt2D(ZtZ[Ik,:] @ DinvIplusZtZD, ZtZ[Ik,:], p, p)

  # Obtain ZtZmat
  RkSum = ZtZmat - secondTerm

  # Multiply by duplication matrices and save
  if not vec:
    covdldDdldsigma2 = 1/(2*sigma2) * invDupMatdict[k] @ mat2vec2D(RkSum)
  else:
    covdldDdldsigma2 = 1/(2*sigma2) * mat2vec2D(RkSum)  
  
  return(covdldDdldsigma2,ZtZmat)



# ============================================================================
#
# Commented out below is an older version of the above code. This has been 
# left here in case it has any use for future development.
#
# ============================================================================
# def get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False):
#  
#   # Sum of R_(k, j) over j
#   RkSum = np.zeros(nparams[k],nparams[k])
#
#   for j in np.arange(nlevels[k]):
#
#     # Get the indices for the kth factor jth level
#     Ikj = faclev_indices2D(k, j, nlevels, nparams)
#
#     # Work out R_(k, j)
#     Rkj = ZtZ[np.ix_(Ikj,Ikj)] - forceSym2D(ZtZ[Ikj,:] @ DinvIplusZtZD @ ZtZ[:,Ikj])
#
#     # Add together
#     RkSum = RkSum + Rkj
#
#   # Multiply by duplication matrices and save
#   if not vec:
#     covdldDdldsigma2 = 1/(2*sigma2) * invDupMatdict[k] @ mat2vec2D(RkSum)
#   else:
#     covdldDdldsigma2 = 1/(2*sigma2) * mat2vec2D(RkSum)  
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
# - `nparams`: A vector containing the number of parameters for each factor,
#              e.g. `nlevels=[2,1]` would mean the first factor has 2 
#              parameters and the second factor has 1 parameter.
# - `ZtZ`: Z transpose multiplied by Z.
# - `DinvIplusZtZD`: D(I+Z'ZD)^(-1) in the above notation.
# - `invDupMatdict`: A dictionary of inverse duplication matrices such that 
#                    `invDupMatdict[k]` = DupMat_k^+
# - `vec` (optional): This is a boolean value which by default is false. If
#                     True it gives the update vector for vec (i.e.
#                     duplicates included), otherwise it gives the update
#                     vector for vech.
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
def get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, perm=None, vec=False):

  # Get the indices for the factors 
  Ik1 = fac_indices2D(k1, nlevels, nparams)
  Ik2 = fac_indices2D(k2, nlevels, nparams)

  # Work out R_(k1,k2)
  Rk1k2 = ZtZ[np.ix_(Ik1,Ik2)] - (ZtZ[Ik1,:] @ DinvIplusZtZD @ ZtZ[:,Ik2])

  # Work out block sizes
  p = np.array([nparams[k1],nparams[k2]])

  # Obtain permutation
  RkRSum,perm=sumAijKronBij2D(Rk1k2, Rk1k2, p, perm)
    
  # Multiply by duplication matrices and save
  if not vec:
    covdldDk1dldk2 = 1/2 * invDupMatdict[k1] @ RkRSum @ invDupMatdict[k2].transpose()
  else:
    covdldDk1dldk2 = 1/2 * RkRSum 

  
  # Return the result
  return(covdldDk1dldk2,perm)


# ============================================================================
#
# Commented out below is an older version of the above code. This has been 
# left here in case it has any use for future development.
#
# ============================================================================
# def get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False):
#  
#   # Sum of R_(k1, k2, i, j) kron R_(k1, k2, i, j) over i and j 
#   for i in np.arange(nlevels[k1]):
#
#     for j in np.arange(nlevels[k2]):
#      
#       # Get the indices for the k1th factor jth level
#       Ik1i = faclev_indices2D(k1, i, nlevels, nparams)
#       Ik2j = faclev_indices2D(k2, j, nlevels, nparams)
#      
#       # Work out R_(k1, k2, i, j)
#       Rk1k2ij = ZtZ[np.ix_(Ik1i,Ik2j)] - (ZtZ[Ik1i,:] @ DinvIplusZtZD @ ZtZ[:,Ik2j])
#      
#       # Work out Rk1k2ij kron Rk1k2ij
#       RkRt = np.kron(Rk1k2ij,Rk1k2ij)
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
#     covdldDk1dldk2 = 1/2 * invDupMatdict[k1] @ RkRtSum @ invDupMatdict[k2].transpose()
#   else:
#     covdldDk1dldk2 = 1/2 * RkRtSum 
#
#  
#   # Return the result
#   return(covdldDk1dldk2)
# ============================================================================


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
        if not perm is None:
          F2=cholmod.symbolic(M,p=perm)
        else:
          F2=cholmod.symbolic(M)
          
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
#
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












    # to document


def elimMat2D(n):

    # Work out indices of lower triangular matrix
    tri_row, tri_col = np.tril_indices(n)

    # Translate these into the column indices we need
    elim_col = np.sort(tri_col*n+tri_row)

    # The row indices are just 1 to n(n+1)/2
    elim_row = np.arange(n*(n+1)//2)

    # We need to put ones in
    elim_dat = np.ones(n*(n+1)//2)

    # Construct the elimination matrix
    elim=scipy.sparse.csr_matrix((elim_dat,(elim_row,elim_col)))

    # Return 
    return(elim)

# ============================================================================
#
# This function maps a vector of the elements of the lower half of a
# lower triangular matrix stacked column-wise to the vector of all elements
# of the matrix.
#
# ============================================================================
def vechTri2mat2D(vech):

    # Return lower triangular
    return(np.tril(vech2mat2D(vech)))

# ============================================================================
#
# This function takes in a (lower triangular, square) matrix and 
# half-vectorizes it (i.e. transforms it to a vector of each of the columns 
# of the matrix, below and including the diagonal, stacked on top of one
# another).
#
# Developer Note: This function is currently a wrapper for mat2vech2D. The 
# reason for this is that, conceptually, both functions return the lower half
# of the input matrix as a vector. However, I distinguished between the two,
# firstly so the steps in each algorithm are more readable and secondly, as
# the functions should expect different inputs. mat2vech2D expects a
# symmetric matrix whilst mat2vechTri2D expects a lower triangular matrix. If
# either of these implementations change in future, it may be useful to have 
# noted the distinction between these functions in the code.
#
# ============================================================================
def mat2vechTri2D(mat):

    # Return vech
    return(mat2vech2D(mat))

