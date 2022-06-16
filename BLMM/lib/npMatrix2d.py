import numpy as np
import scipy.sparse
from scipy import stats

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
    
    # The block is nraneffs by nraneffs
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
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
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
def recursiveInverse2D(M, nraneffs, nlevels):
  
  # Check if we have a matrix we can partition into more than 1 block
  if len(nraneffs) > 1:
  
    # Work out qc (current q)
    qc = nraneffs[-1]*nlevels[-1]
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
    if nraneffs[:-1].shape[0] > 1:

      Ainv = scipy.sparse.csr_matrix(recursiveInverse2D(A, nraneffs[:-1], nlevels[:-1])).toarray()

    else:

      #Ainv = blockInverse(A, nraneffs[0], nlevels[0]) - much slower
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
# This function generates a duplication matrix of size n^2 by n(n+1)/2,
# which maps vec(X) to vechTri(X) for any lower triangular n by n matrix X,
# where vechTri(X) is the vector of the lower triangular elements of X, 
# reading downwards before to the right.
#
# ============================================================================
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
# This function generates the inverse duplication matrix of size n(n+1)/2
# by n^2, which maps vec(X) to vech(X) for any symmetric n by n matrix X.
#
# ============================================================================
def invDupMat2D(n):
  
  # Make vech of 1:(n(n+1)/2)
  vech = np.arange(n*(n+1)/2)
  
  # Convert to vec
  vec = np.int32(vech2vec2D(vech))
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
# - `Ik`: The indices of the columns of Z corresponding to factor k.
#
# *(k is zero indexed)
#
# ============================================================================
def fac_indices2D(k, nlevels, nraneffs):

  # Get indices for all factors
  allInds = np.concatenate((np.array([0]),np.cumsum(nlevels*nraneffs)))

  # Work out the first index
  start = allInds[k]

  # Work out the last index
  end = allInds[k+1]
  
  return(np.arange(start, end))

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
# - `Ikj`: The indices of the columns of Z corresponding to factor k
#          level j.
#
# *(k and j are both zero indexed)
#
# ============================================================================
def faclev_indices2D(k, j, nlevels, nraneffs):

  # Work out the starting point of the indices
  start = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
  
  # work out the end point of the indices
  end = start + nraneffs[k]
  
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
# - `ZtZ`: The Z matrix transposed and then multiplied by itself (Z'Z in the
#          above notation).
# - `Zte`: The Z matrix transposed and then multiplied by the OLS residuals
#          (Z'e=Z'(Y-X\beta) in the above notation).
# - `sigma2`: The OLS estimate of \sigma^2 (\sigma^2 in the above notation).
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
def initDk2D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict):
  
  # Initalize to zeros
  ZkjtZkj = np.zeros((nraneffs[k],nraneffs[k]))

  # First we work out the derivative we require.
  for j in np.arange(nlevels[k]):
    
    Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

    # Work out Z_(k, j)'Z_(k, j)
    ZkjtZkj = ZkjtZkj + ZtZ[np.ix_(Ikj,Ikj)]
    
  # Work out block size
  qk = nraneffs[k]
  p = np.array([qk,1])

  # Get indices
  Ik = fac_indices2D(k, nlevels, nraneffs)

  # Work out the sum of Z_{(k,j)}'ee'Z_{(k,j)}
  ZteetZ = sumAijBijt2D(Zte[Ik,:],Zte[Ik,:],p,p)

  # Add first \sigma^{-2}Z'ee'Z - Z_(k,j)'Z_(k,j)
  invSig2ZteetZminusZtZ = 1/sigma2*ZteetZ - ZkjtZkj
  
  # Second we need to work out the double sum of Z_(k,j)'Z_(k,j)
  p = np.array([nraneffs[k],nraneffs[k]])

  # Get sum of Z_{(k,j)}'Z_{(k,j)} kron Z_{(k,j)}'Z_{(k,j)}
  ZtZkronZtZ,_ = sumAijKronBij2D(ZtZ[np.ix_(Ik,Ik)], ZtZ[np.ix_(Ik,Ik)], p, perm=None)

  # Work out information matrix
  infoMat = dupMatTdict[k] @ ZtZkronZtZ @ dupMatTdict[k].transpose()

  # Work out the final term.
  Dkest = vech2mat2D(np.linalg.solve(infoMat, dupMatTdict[k] @ mat2vec2D(invSig2ZteetZminusZtZ))) 
  
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
  if not np.all(np.linalg.eigvals(D)>=0):
  
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
#
# The below function takes in a covariance matrix D and finds a projection of 
# it onto the space of positive definite matrices D_+'. It uses the following
# method taken from Demidenko (2012), page 105:
#
# If D is has eigenvalue decomposition D=P\Lambda P' it's projection into D_+'
# is defined by the matrix below:
#
# Dhat_+ = P\Lambda_+P'
#
# Where \Lambda_+ is defined by the elementwise maximum of \Lambda and 0; i.e.
# \Lambda_+(i,j) = max(\Lambda_+(i,j),1e-6).
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
# - `D_nnd`: A projection of D onto the space of positive definite matrices 
#            D_+.
#
# ============================================================================
def makeDpd2D(D):
  
  # Check if we have negative or zero valued eigenvalues
  if not np.all(np.linalg.eigvals(D)>0):
  
    # If we have negative eigenvalues
    eigvals,eigvecs = np.linalg.eigh(D)
    
    # Work out elementwise max of lambda and 0
    lamplus = np.diag(np.maximum(eigvals,1e-6))
    
    # Work out D+
    D_pd = eigvecs @ lamplus @ np.linalg.inv(eigvecs)
    
  else:
    
    # D is already positive definite in this case
    D_pd = D
    
  return(D_pd)

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
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
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
def get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD, ZtZmat=None):

  # We only need calculate this once across all iterations
  if ZtZmat is None:

    # Instantiate to zeros
    ZtZmat = np.zeros(nraneffs[k],nraneffs[k])

    for j in np.arange(nlevels[k]):

      # Get the indices for the kth factor jth level
      Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

      # Work out Z_(k,j)'Z_(k,j)
      ZtZterm = ZtZ[np.ix_(Ikj,Ikj)]

      # Add together
      ZtZmat = ZtZmat + ZtZterm

  # Get the indices for the factors 
  Ik = fac_indices2D(k, nlevels, nraneffs)

  # Work out lk
  lk = nlevels[k]

  # Work out block size
  qk = nraneffs[k]
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
# def get_dldDk2D_old(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD):
#
#   # Initalize the derivative to zeros
#   dldDk = np.zeros((nraneffs[k],nraneffs[k]))
#
#   # For each level j we need to add a term
#   for j in np.arange(nlevels[k]):
#
#     # Get the indices for the kth factor jth level
#     Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
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
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `ZtZ`: Z transpose multiplied by Z.
# - `DinvIplusZtZD`: D(I+Z'ZD)^(-1) in the above notation.
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
def get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=False, ZtZmat=None):

  # We only need calculate this once across all iterations
  if ZtZmat is None:

    # Instantiate to zeros
    ZtZmat = np.zeros(nraneffs[k],nraneffs[k])

    for j in np.arange(nlevels[k]):

      # Get the indices for the kth factor jth level
      Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

      # Work out R_(k, j)
      ZtZterm = ZtZ[np.ix_(Ikj,Ikj)]

      # Add together
      ZtZmat = ZtZmat + ZtZterm

  # Get the indices for the factors 
  Ik = fac_indices2D(k, nlevels, nraneffs)

  # Work out lk
  lk = nlevels[k]

  # Work out block size
  q = np.sum(nlevels*nraneffs)
  qk = nraneffs[k]
  p = np.array([qk,q])

  # Work out the second term
  secondTerm = sumAijBijt2D(ZtZ[Ik,:] @ DinvIplusZtZD, ZtZ[Ik,:], p, p)

  # Obtain sum of Rk
  RkSum = ZtZmat - secondTerm

  # Multiply by duplication matrices and save
  if not vec:
    covdldDdldsigma2 = 1/(2*sigma2) * dupMatTdict[k] @ mat2vec2D(RkSum)
  else:
    covdldDdldsigma2 = 1/(2*sigma2) * mat2vec2D(RkSum)  
  
  return(covdldDdldsigma2,ZtZmat)



# ============================================================================
#
# Commented out below is an older version of the above code. This has been 
# left here in case it has any use for future development.
#
# ============================================================================
# def get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=False):
#  
#   # Sum of R_(k, j) over j
#   RkSum = np.zeros(nraneffs[k],nraneffs[k])
#
#   for j in np.arange(nlevels[k]):
#
#     # Get the indices for the kth factor jth level
#     Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
#
#     # Work out R_(k, j)
#     Rkj = ZtZ[np.ix_(Ikj,Ikj)] - forceSym2D(ZtZ[Ikj,:] @ DinvIplusZtZD @ ZtZ[:,Ikj])
#
#     # Add together
#     RkSum = RkSum + Rkj
#
#   # Multiply by duplication matrices and save
#   if not vec:
#     covdldDdldsigma2 = 1/(2*sigma2) * dupMatTdict[k] @ mat2vec2D(RkSum)
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
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `ZtZ`: Z transpose multiplied by Z.
# - `DinvIplusZtZD`: D(I+Z'ZD)^(-1) in the above notation.
# - `dupMatTdict`: A dictionary of transpose duplication matrices such that 
#                    `dupMatTdict[k]` = DupMat_k'
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
def get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, perm=None, vec=False):

  # Get the indices for the factors 
  Ik1 = fac_indices2D(k1, nlevels, nraneffs)
  Ik2 = fac_indices2D(k2, nlevels, nraneffs)

  # Work out R_(k1,k2)
  Rk1k2 = ZtZ[np.ix_(Ik1,Ik2)] - (ZtZ[Ik1,:] @ DinvIplusZtZD @ ZtZ[:,Ik2])

  # Work out block sizes
  p = np.array([nraneffs[k1],nraneffs[k2]])

  # Obtain permutation
  RkRSum,perm=sumAijKronBij2D(Rk1k2, Rk1k2, p, perm)
    
  # Multiply by duplication matrices and save
  if not vec:
    covdldDk1dldk2 = 1/2 * dupMatTdict[k1] @ RkRSum @ dupMatTdict[k2].transpose()
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
# def get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=False):
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
#     covdldDk1dldk2 = 1/2 * dupMatTdict[k1] @ RkRtSum @ dupMatTdict[k2].transpose()
#   else:
#     covdldDk1dldk2 = 1/2 * RkRtSum 
#
#  
#   # Return the result
#   return(covdldDk1dldk2)
# ============================================================================


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
# - `n`: The number of observations
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
def get_resms2D(YtX, YtY, XtX, beta, n, p):

  # Work out e'e
  ete = ssr2D(YtX, YtY, XtX, beta)

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
# - `XtX`: X transpose multiplied by X (X'X in the previous notation).
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
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
def get_covB2D(XtX, XtZ, DinvIplusZtZD, sigma2):

  # Work out X'V^{-1}X = X'X - X'ZD(I+Z'ZD)^{-1}Z'X
  XtinvVX = XtX - XtZ @ DinvIplusZtZD @ XtZ.transpose()

  # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
  covB = np.linalg.pinv(XtinvVX)

  # Calculate sigma^2(X'V^{-1}X)^(-1)
  covB = sigma2*covB

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
# - `XtX`: X transpose multiplied by X (X'X in the previous notation).
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
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
def get_varLB2D(L, XtX, XtZ, DinvIplusZtZD, sigma2):

  # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
  varLB = L @ get_covB2D(XtX, XtZ, DinvIplusZtZD, sigma2) @ L.transpose()

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
# - `df`: The denominator degrees of freedom of the F statistic.
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
def get_R22D(L, F, df):

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
# - `XtX`: X transpose multiplied by X (X'X in the previous notation).
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
# - `beta`: The estimate of the fixed effects parameters.
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
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
def get_T2D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2):

  # Work out the rank of L
  rL = np.linalg.matrix_rank(L)

  # Work out Lbeta
  LB = L @ beta

  # Work out se(T)
  varLB = get_varLB2D(L, XtX, XtZ, DinvIplusZtZD, sigma2)

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
# - `XtX`: X transpose multiplied by X (X'X in the previous notation).
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
# - `beta`: The estimate of the fixed effects parameters.
# - `sigma2`: The fixed effects variance (\sigma^2 in the previous notation).
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
def get_F2D(L, XtX, XtZ, DinvIplusZtZD, betahat, sigma2):

  # Work out the rank of L
  rL = np.linalg.matrix_rank(L)

  # Work out Lbeta
  LB = L @ betahat

  # Work out se(F)
  varLB = get_varLB2D(L, XtX, XtZ, DinvIplusZtZD, sigma2)

  # Work out F
  F = LB.transpose() @ np.linalg.pinv(varLB) @ LB/rL

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
# - `df`: The degrees of freedom of the T statistic.
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
def T2P2D(T,df,minlog):

  # Do this seperately for >0 and <0 to avoid underflow
  if T < 0:
    P = -np.log10(1-stats.t.cdf(T, df))
  else:
    P = -np.log10(stats.t.cdf(-T, df))

  # Remove infs
  if np.logical_and(np.isinf(P), P<0):
    P = minlog

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
# - `df_denom`: The denominator degrees of freedom of the F statistic.
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
def F2P2D(F, L, df_denom, minlog):
  
  # Get the rank of L
  df_num = np.linalg.matrix_rank(L)

  # Work out P
  P = -np.log10(1-stats.f.cdf(F, df_num, df_denom))

  # Remove infs
  if np.logical_and(np.isinf(P), P<0):
    P = minlog

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
# `get_swdf_T2D` below). 
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast matrix.
# - `D`: The random effects variance-covariance matrix estimate.
# - `sigma2`: The fixed effects variance estimate.
# - `XtX`: X transpose multiplied by X (X'X in the previous notation).
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `ZtX`: Z transpose multiplied by X (Z'X in the previous notation).
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation).
# - `n`: The number of observations
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
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
# - `df`: The Sattherthwaithe degrees of freedom estimate.
#
# ============================================================================
def get_swdf_F2D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs): 

  # L is rL in rank
  rL = np.linalg.matrix_rank(L)

  # Initialize empty sum.
  sum_swdf_adj = 0

  # Loop through first rL rows of L
  for i in np.arange(rL):

    # Work out the swdf for each row of L
    swdf_row = get_swdf_T2D(L[i:(i+1),:], D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs)

    # Work out adjusted df = df/(df-2)
    swdf_adj = swdf_row/(swdf_row-2)

    # Add to running sum
    sum_swdf_adj = sum_swdf_adj + swdf_adj[0]

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
# - `D`: The random effects variance-covariance matrix estimate.
# - `sigma2`: The fixed effects variance estimate.
# - `XtX`: X transpose multiplied by X (X'X in the previous notation).
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `ZtX`: Z transpose multiplied by X (Z'X in the previous notation).
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation).
# - `n`: The number of observations.
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
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
# - `df`: The Sattherthwaithe degrees of freedom estimate.
#
# ============================================================================
def get_swdf_T2D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs): 

  # Get D(I+Z'ZD)^(-1)
  DinvIplusZtZD = np.linalg.solve(np.eye(ZtZ.shape[1]) + D @ ZtZ, D)

  # Get S^2 (= Var(L\beta))
  S2 = get_varLB2D(L, XtX, XtZ, DinvIplusZtZD, sigma2)
  
  # Get derivative of S^2
  dS2 = get_dS22D(nraneffs, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2)

  # Get Fisher information matrix
  InfoMat = get_InfoMat2D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ)

  # Calculate df estimator
  df = 2*(S2**2)/(dS2.transpose() @ np.linalg.solve(InfoMat, dS2))

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
# - `XtX`: X transpose multiplied by X (X'X in the previous notation).
# - `XtZ`: X transpose multiplied by Z (X'Z in the previous notation).
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation).
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
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
def get_dS22D(nraneffs, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2):

  # ZtX
  ZtX = XtZ.transpose()

  # Calculate X'V^{-1}X=X'(I+ZDZ')^{-1}X=X'X-X'Z(I+DZ'Z)^{-1}DZ'X
  XtiVX = XtX - XtZ @  DinvIplusZtZD @ ZtX

  # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
  dS2 = np.zeros((1+np.int32(np.sum(nraneffs*(nraneffs+1)/2)),1))

  # Work out indices for each start of each component of vector 
  # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
  DerivInds = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
  DerivInds = np.insert(DerivInds,0,1)

  # Work of derivative wrt to sigma^2
  dS2dsigma2 = L @ np.linalg.pinv(XtiVX) @ L.transpose()

  # Add to dS2
  dS2[0:1,0] = dS2dsigma2.reshape(dS2[0:1,0].shape)

  # Now we need to work out ds2dVech(Dk)
  for k in np.arange(len(nraneffs)):

    # Initialize an empty zeros matrix
    dS2dvechDk = np.zeros((np.int32(nraneffs[k]*(nraneffs[k]+1)/2),1))

    for j in np.arange(nlevels[k]):

      # Get the indices for this level and factor.
      Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
              
      # Work out Z_(k,j)'Z
      ZkjtZ = ZtZ[Ikj,:]

      # Work out Z_(k,j)'X
      ZkjtX = ZtX[Ikj,:]

      # Work out Z_(k,j)'V^{-1}X
      ZkjtiVX = ZkjtX - ZkjtZ @ DinvIplusZtZD @ ZtX

      # Work out the term to put into the kronecker product
      # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
      K = ZkjtiVX @ np.linalg.pinv(XtiVX) @ L.transpose()
      
      # Sum terms
      dS2dvechDk = dS2dvechDk + dupMat2D(nraneffs[k]).toarray().transpose() @ mat2vec2D(np.kron(K,K.transpose()))

    # Multiply by sigma^2
    dS2dvechDk = sigma2*dS2dvechDk

    # Add to dS2
    dS2[DerivInds[k]:DerivInds[k+1],0] = dS2dvechDk.reshape(dS2[DerivInds[k]:DerivInds[k+1],0].shape)

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
# - `DinvIplusZtZD`: The product D(I+Z'ZD)^(-1).
# - `sigma2`: The fixed effects variance estimate.
# - `n`: The total number of observations.
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `ZtZ`: Z transpose multiplied by Z (Z'Z in the previous notation).
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
def get_InfoMat2D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ):

  # Number of random effects, q
  q = np.sum(np.dot(nraneffs,nlevels))

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
  FisherInfoMat = np.zeros((tnp,tnp))
  
  # Covariance of dl/dsigma2
  covdldsigma2 = n/(2*(sigma2**2))
  
  # Add dl/dsigma2 covariance
  FisherInfoMat[0,0] = covdldsigma2

  
  # Add dl/dsigma2 dl/dD covariance
  for k in np.arange(len(nraneffs)):

    # Get covariance of dldsigma and dldD      
    covdldsigma2dD = get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)[0].reshape(FishIndsDk[k+1]-FishIndsDk[k])

    # Assign to the relevant block
    FisherInfoMat[0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigma2dD
    FisherInfoMat[FishIndsDk[k]:FishIndsDk[k+1],0:1] = FisherInfoMat[0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose()
  
  # Add dl/dD covariance
  for k1 in np.arange(len(nraneffs)):

    for k2 in np.arange(k1+1):

      IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
      IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

      # Get covariance between D_k1 and D_k2 
      covdldDk1dDk2 = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)[0]

      # Add to FImat
      FisherInfoMat[np.ix_(IndsDk1, IndsDk2)] = covdldDk1dDk2
      FisherInfoMat[np.ix_(IndsDk2, IndsDk1)] = FisherInfoMat[np.ix_(IndsDk1, IndsDk2)].transpose()


  # Return result
  return(FisherInfoMat)
