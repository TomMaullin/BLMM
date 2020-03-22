import os
import sys
import numpy as np
import cvxopt
import pandas as pd
import os
import time
import scipy.sparse
import scipy.sparse.linalg
import sys

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.tools2d import *

# =============================================================================
# This file contains all unit tests for the functions given in the tools2D.py
# file.
#
# Author: Tom Maullin
# Last edited: 21/03/2020
#
# =============================================================================


# =============================================================================
#
# The below function generates a random testcase according to the linear mixed
# model:
#
#   Y = X\beta + Zb + \epsilon
#
# Where b~N(0,D) and \epsilon ~ N(0,\sigma^2 I)
#
# -----------------------------------------------------------------------------
#
# It takes the following inputs:
#
# -----------------------------------------------------------------------------
#
#   - n (optional): Number of subjects. If not provided, a random n will be
#                   selected between 800 and 1200.
#   - p (optional): Number of fixed effects parameters. If not provided, a
#                   random p will be selected between 2 and 10 (an intercept is
#                   automatically included).
#   - nlevels (optional): A vector containing the number of levels for each
#                         random factor, e.g. `nlevels=[3,4]` would mean the
#                         first factor has 3 levels and the second factor has
#                         4 levels. If not provided, default values will be
#                         between 8 and 40.
#   - nparams (optional): A vector containing the number of parameters for each
#                         factor, e.g. `nlevels=[2,1]` would mean the first
#                         factor has 2 parameters and the second factor has 1
#                         parameter. If not provided, default values will be
#                         between 2 and 5.
#
# -----------------------------------------------------------------------------
#
# And gives the following outputs:
#
# -----------------------------------------------------------------------------
#
#   - X: A fixed effects design matrix of dimensions (n x p) including a random
#        intercept column (the first column).
#   - Y: A response vector of dimension (n x 1).
#   - Z: A random effects design matrix of size (n x q) where q is equal to the
#        product of nlevels and nparams.
#   - nlevels: A vector containing the number of levels for each random factor,
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
#   - nparams: A vector containing the number of parameters for each factor,
#              e.g. `nlevels=[2,1]` would mean the first factor has 2
#              parameters and the second factor has 1 parameter.
#   - beta: The true values of beta used to simulate the response vector.
#   - sigma2: The true value of sigma2 used to simulate the response vector.
#   - D: The random covariance matrix used to simulate b and the response vector.
#   - b: The random effects vector used to simulate the response vector.
#
# -----------------------------------------------------------------------------
def genTestData(n=None, p=None, nlevels=None, nparams=None):

    # Check if we have n
    if n is None:

        # If not generate a random n
        n = np.random.randint(800,1200)
    
    # Check if we have p
    if p is None:

        # If not generate a random p
        p = np.random.randint(2,10)

    # Work out number of random factors.
    if nlevels is None and nparams is None:

        # If we have neither nlevels or nparams, decide on a number of
        # random factors, r.
        r = np.random.randint(2,4)

    elif nlevels is None:

        # Work out number of random factors, r
        r = np.shape(nparams)[0]

    else:

        # Work out number of random factors, r
        r = np.shape(nlevels)[0]

    # Check if we need to generate nlevels.
    if nlevels is None:
        
        # Generate random number of levels.
        nlevels = np.random.randint(8,40,r)

    # Check if we need to generate nparams.
    if nparams is None:
        
        # Generate random number of levels.
        nparams = np.random.randint(2,5,r)

    # Generate random X.
    X = np.random.randn(n,p)
    
    # Make the first column an intercept
    X[:,0]=1

    # Generate beta (used integers to make test results clear).
    beta = np.random.randint(-10,10,p)

    # Create Z
    # We need to create a block of Z for each level of each factor
    for i in np.arange(r):

        Zdata_factor = np.random.randn(n,nparams[i])

        if i==0:

            #The first factor should be block diagonal, so the factor indices are grouped
            factorVec = np.repeat(np.arange(nlevels[i]), repeats=np.floor(n/max(nlevels[i],1)))

            if len(factorVec) < n:

                # Quick fix incase rounding leaves empty columns
                factorVecTmp = np.zeros(n)
                factorVecTmp[0:len(factorVec)] = factorVec
                factorVecTmp[len(factorVec):n] = nlevels[i]-1
                factorVec = np.int64(factorVecTmp)


                # Crop the factor vector - otherwise have a few too many
                factorVec = factorVec[0:n]

                # Give the data an intercept
                Zdata_factor[:,0]=1

        else:

            # The factor is randomly arranged across subjects
            factorVec = np.random.randint(0,nlevels[i],size=n) 

        # Build a matrix showing where the elements of Z should be
        indicatorMatrix_factor = np.zeros((n,nlevels[i]))
        indicatorMatrix_factor[np.arange(n),factorVec] = 1

        # Need to repeat for each parameter the factor has 
        indicatorMatrix_factor = np.repeat(indicatorMatrix_factor, nparams[i], axis=1)

        # Enter the Z values
        indicatorMatrix_factor[indicatorMatrix_factor==1]=Zdata_factor.reshape(Zdata_factor.shape[0]*Zdata_factor.shape[1])

        # Make sparse
        Zfactor = scipy.sparse.csr_matrix(indicatorMatrix_factor)

        # Put all the factors together
        if i == 0:
            Z = Zfactor
        else:
            Z = scipy.sparse.hstack((Z, Zfactor))

    # Convert Z to dense
    Z = Z.toarray()

    # Make random beta
    beta = np.random.randint(-5,5,p).reshape(p,1)

    # Make random sigma2
    sigma2 = 0.5*np.random.randn()**2

    # Make epsilon.
    epsilon = sigma2*np.random.randn(n).reshape(n,1)

    # Make random D
    Ddict = dict()
    Dhalfdict = dict()
    for k in np.arange(r):
      
        # Create a random matrix
        randMat = np.random.randn(nparams[k],nparams[k])

        # Record it as D^{1/2}
        Dhalfdict[k] = randMat

        # Work out D = D^{1/2} @ D^{1/2}'
        Ddict[k] = randMat @ randMat.transpose()

    # Matrix version
    D = np.array([])
    Dhalf = np.array([])
    for i in np.arange(r):
      
        for j in np.arange(nlevels[i]):
        
            if i == 0 and j == 0:

                D = Ddict[i]
                Dhalf = Dhalfdict[i]

            else:

                D = scipy.linalg.block_diag(D, Ddict[i])
                Dhalf = scipy.linalg.block_diag(Dhalf, Dhalfdict[i])


    # Make random b
    q = np.sum(nlevels*nparams)
    b = np.random.randn(q).reshape(q,1)

    # Give b the correct covariance structure
    b = Dhalf @ b

    # Generate the response vector
    Y = X @ beta + Z @ b + epsilon

    # Return values
    return(Y,X,Z,nlevels,nparams,beta,sigma2,b,D)


# =============================================================================
#
# The below function generates the product matrices from matrices X, Y and Z.
#
# -----------------------------------------------------------------------------
#
# It takes as inputs:
#
# -----------------------------------------------------------------------------
#
#  - `X`: The design matrix of dimension n times p.
#  - `Y`: The response vector of dimension n times 1.
#  - `Z`: The random effects design matrix of dimension n times q.
#
# -----------------------------------------------------------------------------
#
# It returns as outputs:
#
# -----------------------------------------------------------------------------
#
#  - `XtX`: X transposed multiplied by X.
#  - `XtY`: X transposed multiplied by Y.
#  - `XtZ`: X transposed multiplied by Z.
#  - `YtX`: Y transposed multiplied by X.
#  - `YtY`: Y transposed multiplied by Y.
#  - `YtZ`: Y transposed multiplied by Z.
#  - `ZtX`: Z transposed multiplied by X.
#  - `ZtY`: Z transposed multiplied by Y.
#  - `ZtZ`: Z transposed multiplied by Z.
#
# =============================================================================
def prodMats(Y,Z,X):

    # Work out the product matrices
    XtX = X.transpose() @ X
    XtY = X.transpose() @ Y
    XtZ = X.transpose() @ Z
    YtX = XtY.transpose()
    YtY = Y.transpose() @ Y
    YtZ = Y.transpose() @ Z
    ZtX = XtZ.transpose()
    ZtY = YtZ.transpose()
    ZtZ = Z.transpose() @ Z

    # Return product matrices
    return(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ)


# =============================================================================
#
# The below function tests the function `mat2vec2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_mat2vec2D():

    # Example matrix
    matrix = np.arange(9).reshape([3,3])

    # Expected answer
    expected = np.array([0,3,6,1,4,7,2,5,8]).reshape([9,1])

    # Test
    testVal = np.allclose(mat2vec2D(matrix), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mat2vec2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `mat2vech2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_mat2vech2D():

    # Example matrix (must be symmetric)
    matrix = np.arange(16).reshape([4,4])
    matrix = matrix @ matrix.transpose()

    # Expected result
    expected = np.array([14,38,62,86,126,214,302,366,518,734]).reshape([10,1])

    # Test
    testVal = np.allclose(mat2vech2D(matrix), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mat2vech2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `vec2mat2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_vec2mat2D():

    # Generate vector
    vec = np.arange(16).reshape([16,1])

    # Expected result
    expected = np.array([[0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]])

    # Test
    testVal = np.allclose(vec2mat2D(vec), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: vec2mat2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `vech2mat2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_vech2mat2D():

    # Generate vector
    vec = np.arange(10).reshape([10,1])

    # Expected result
    expected = np.array([[0,1,2,3],[1,4,5,6],[2,5,7,8],[3,6,8,9]])

    # Test
    testVal = np.allclose(vech2mat2D(vec), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: vech2mat2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)



# =============================================================================
#
# The below function tests the function `vec2vech2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_vec2vech2D():

    # Generate vector
    vec = np.array([0,1,2,1,3,4,2,4,5]).reshape([9,1])

    # Expected result
    expected = np.arange(6).reshape([6,1])

    # Test
    testVal = np.allclose(vec2vech2D(vec), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: vec2vech2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)



# =============================================================================
#
# The below function tests the function `vech2vec2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_vech2vec2D():

    # Generate vector
    vec = np.arange(6).reshape([6,1])

    # Expected result
    expected = np.array([0,1,2,1,3,4,2,4,5]).reshape([9,1])
    
    # Test
    testVal = np.allclose(vech2vec2D(vec), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: vech2vec2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)



# =============================================================================
#
# The below function tests the function `blockInverse2D`. It does this by  
# generating a random block matrix input and checking against numpy.
#
# =============================================================================
def test_blockInverse2D():

    # Random number of blocks
    numBlocks = np.random.randint(700,1000)

    # Random block size
    blockSize = np.random.randint(2,6)

    # Generate a random block matrix
    for i in np.arange(numBlocks):

        # Generate random current block
        currentBlock = np.random.randn(blockSize,blockSize)

        # Stack the blocks diagonally
        if i==0:

            blockMat = currentBlock

        else:

            blockMat = scipy.linalg.block_diag(blockMat, currentBlock)

    # Make it sparse.
    blockMat_sp = scipy.sparse.csc_matrix(blockMat)

    # Test the function
    invBlockMat_test = blockInverse2D(blockMat_sp,blockSize).toarray()
    
    # Expected result
    invBlockMat_expected = np.linalg.inv(blockMat)
    
    # Test
    testVal = np.allclose(invBlockMat_test, invBlockMat_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: blockInverse2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `recursiveInverse2D`. It does this by
# simulating some test data and checking performance against numpy.
#
# =============================================================================
def test_recursiveInverse2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()

    # Work out q
    q = np.sum(nlevels*nparams)

    # Work out I+Z'ZD
    ZtZ = Z.transpose() @ Z 

    # Sparse version
    ZtZ_sp = scipy.sparse.csr_matrix(ZtZ)

    # Recursive inverse
    ZtZinv_test = recursiveInverse2D(ZtZ_sp, nparams, nlevels)

    # Regular inverse
    ZtZinv_expected = np.linalg.inv(ZtZ)
    
     # Test
    testVal = np.allclose(ZtZinv_test, ZtZinv_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: recursiveInverse2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `dupMat2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_dupMat2D():

    # Expected duplication matrix for 3 by 3 matrices
    expected = np.array([[1,0,0,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,1,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,0],
                         [0,0,1,0,0,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]])

    # Test
    testVal = np.allclose(dupMat2D(3).toarray(), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: dupMat2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `invDupMat2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_invDupMat2D():

    # Expected inverse duplication matrix for 3 by 3 matrices
    expected = np.array([[1,0,0,0,0,0],
                         [0,0.5,0,0,0,0],
                         [0,0,0.5,0,0,0],
                         [0,0.5,0,0,0,0],
                         [0,0,0,1,0,0],
                         [0,0,0,0,0.5,0],
                         [0,0,0.5,0,0,0],
                         [0,0,0,0,0.5,0],
                         [0,0,0,0,0,1]]).transpose()

    # Test
    testVal = np.allclose(invDupMat2D(3).toarray(), expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: invDupMat2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `comMat2D`. It does this by generating
# random matrices and checking the commutation matrix has the desired effect on
# said matrices.
#
# =============================================================================
def test_comMat2D():

    # Generate 2 random dimensions
    a = np.random.randint(100,500)
    b = np.random.randint(100,500)

    # Test commutation matrix
    K = comMat2D(a,b)
    
    # Random matrix
    A = np.random.randn(a,b)

    # Check the commutation matrix does as expected
    testVal = np.allclose(mat2vec2D(A.transpose()),K @ mat2vec2D(A))

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: comMat2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)



# =============================================================================
#
# The below function tests the function `permOfIkKkI2D`. It does this by 
# generating a random example and checking the permutation vector gives the
# same as  what we would expect if we had used the permutation matrices.
#
# =============================================================================
def test_permOfIkKkI2D():

    # First generate random matrix sizes
    m = np.random.randint(1,10)
    n1=np.random.randint(1,10)
    n2=np.random.randint(1,10)
    k1=np.random.randint(1,10)
    k2=np.random.randint(1,10)

    # Generate a random matrix of appropriate size for testing
    testMat = np.random.randn(k1*k2*n1*n2,m)

    # For testing work out K
    K = comMat2D(n1, k2).toarray()

    # Work out I kron K kron I
    IkKkI = np.kron(np.kron(np.eye(k1), K), np.eye(n2))

    # Work out the permutation which supposedly represents IkKkI
    perm = permOfIkKkI2D(k1,k2,n1,n2)

    # Check that permuting using perm is the same as right multiplying
    # by IkKkI
    testVal = np.allclose(IkKkI @ testMat, testMat[perm,:])

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: permOfIkKkI2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `block2stacked2D`. It does this by 
# checking the function against a predefined example.
#
# =============================================================================
def test_block2stacked2D():

    # Example test case
    matrix = np.arange(24).reshape(4,6)

    # Partition size
    pMatrix = np.array([2,3])

    # Expected output
    expected = np.array([[0,1,2],
                         [6,7,8],
                         [3,4,5],
                         [9,10,11],
                         [12,13,14],
                         [18,19,20],
                         [15,16,17],
                         [21,22,23]])

    # Check if the function gives as expected
    testVal = np.allclose(block2stacked2D(matrix,pMatrix),expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: block2stacked2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `mat2vecb`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_mat2vecb2D():

    # Example test case
    matrix = np.arange(24).reshape(4,6)

    # Partition size
    pMatrix = np.array([2,3])

    # Expected output
    expected = np.array([[0,6,1,7,2,8],
                         [3,9,4,10,5,11],
                         [12,18,13,19,14,20],
                         [15,21,16,22,17,23]])

    # Check if the function gives as expected
    testVal = np.allclose(mat2vecb2D(matrix,pMatrix),expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mat2vecb2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `sumAijBijt2D`. It does this by 
# generating random matrices and checks that the function behaves the same as 
# niave calculation of the same quantity.
#
# =============================================================================
def test_sumAijBijt2D():

    # Random number of blocks
    l1 = np.random.randint(1,3000)
    l2 = np.random.randint(1,3000)

    # Random blocksizes (the second dimension must be the same for both)
    n1 = np.random.randint(1,5)
    n1prime = np.random.randint(1,5)
    n2 = np.random.randint(1,5)

    # Save block sizes
    pA = np.array([n1,n2])
    pB = np.array([n1prime,n2])

    # Work out m1, m1' and m2
    m1 = n1*l1
    m1prime = n1prime*l1
    m2 = n2*l2

    # Create random A and B
    A = np.random.randn(m1,m2)
    B = np.random.randn(m1prime,m2)

    # Calculate the answer we would expect independently.
    sumAB = np.zeros((n1,n1prime))

    # Sum over all blocks
    for i in np.arange(m1//n1):
      for j in np.arange(m2//n2):

        # Work out the row-wise chunks
        Aij = A[n1*i:n1*(i+1),n2*j:n2*(j+1)]
        Bij = B[n1prime*i:n1prime*(i+1),n2*j:n2*(j+1)]

        # Perform the summation
        sumAB = sumAB + Aij @ Bij.transpose()

    # Now perform the same calculation using the function.
    sumAB_test = sumAijBijt2D(A, B, pA, pB)

    # Check if the function gives as expected
    testVal = np.allclose(sumAB,sumAB_test)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: sumAijBijt2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `sumAijKronBij2D`. It does this by 
# generating random matrices and checks that the function behaves the same as 
# niave calculation of the same quantity.
#
# =============================================================================
def test_sumAijKronBij2D():

    # Random block sizes
    n1 = np.random.randint(2,10)
    n2 = np.random.randint(2,10)

    # Blocksize
    p = np.array([n1,n2])

    # Random number of blocks
    l1 = np.random.randint(2,30)
    l2 = np.random.randint(2,30)

    # Resultant matrix sizes
    m1 = l1*n1
    m2 = l2*n2

    # Random matrices
    A = np.random.randn(m1,m2)
    B = np.random.randn(m1,m2)

    # Work out the sum manually
    runningSum = np.zeros((n1**2,n2**2))
    t1 = time.time()
    for i in np.arange(l1):
      for j in np.arange(l2):

        Aij = A[n1*i:n1*(i+1),n2*j:n2*(j+1)]
        Bij = B[n1*i:n1*(i+1),n2*j:n2*(j+1)]

        runningSum = runningSum + np.kron(Aij, Bij)

    # Work out the sum using the function
    S, perm = sumAijKronBij2D(A, B, p)

    # Work out the sum using the function (and permutation)
    S2, _ = sumAijKronBij2D(A, B, p, perm)

    # Check if the function gives as expected
    testVal = np.allclose(S, runningSum) and np.allclose(S2, runningSum)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: sumAijKronBij2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `forceSym2D`. It does this by checking 
# the function against a predefined example.
#
# =============================================================================
def test_forceSym2D():

    # Test example
    matrix = np.eye(10)
    matrix[3,2] = 7
    matrix[2,3] = 5

    # Expected output
    expected = np.eye(10)
    expected[3,2] = 6
    expected[2,3] = 6

    # Check if the function gives as expected
    testVal = np.allclose(forceSym2D(matrix),expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: forceSym2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `ssr2D`.  It does this by simulating
# data and checks that the function behaves the same as niave calculation of
# the same quantity.
#
# =============================================================================
def test_ssr2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Work out the sum of squared residuals we would expect
    ssr_expected = (Y - X @ beta).transpose() @ (Y - X @ beta)

    # Work out the ssr the test gives
    ssr_test = ssr2D(YtX, YtY, XtX, beta)

    # Check if the function gives as expected
    testVal = np.allclose(ssr_test,ssr_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: ssr2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `fac_indices2D`. It does this by
# checking the function against a predefined example.
#
# =============================================================================
def test_fac_indices2D():

    # Test nlevels, nparams,k
    k = 2
    nlevels = np.array([3,4,2,8])
    nparams = np.array([1,2,3,4])

    # Expected output
    expected = np.arange(11,17)

    # Check if the function gives as expected
    testVal = np.allclose(fac_indices2D(k,nlevels,nparams),expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: fac_indices2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `faclev_indices2D`. It does this by
# checking the function against a predefined example.
#
# =============================================================================
def test_faclev_indices2D():

    # Test nlevels, nparams,k
    k = 2
    j = 1
    nlevels = np.array([3,4,2,8])
    nparams = np.array([1,2,3,4])

    # Expected output
    expected = np.arange(14,17)

    # Check if the function gives as expected
    testVal = np.allclose(faclev_indices2D(k,j,nlevels,nparams),expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: faclev_indices2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `initBeta2D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_initBeta2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()

    # Calculate OLS estimate using test data.
    expected = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Get the initial beta value
    betahat = initBeta2D(XtX,XtY)

    # Check if the function gives as expected
    testVal = np.allclose(betahat,expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: initBeta2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `initSigma22D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_initSigma22D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Get the initial beta value
    betahat = initBeta2D(XtX,XtY)

    # Calculate OLS estimate using test data.
    expected = (Y - X @ betahat).transpose() @ (Y - X @ betahat)/n

    # Get ete
    ete = ssr2D(YtX, YtY, XtX, betahat)

    # Get the initial sigma2 value
    sigma2hat = initSigma22D(ete,n)

    # Check if the function gives as expected
    testVal = np.allclose(sigma2hat,expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: initSigma22D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `initDk2D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_initDk2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Get the initial beta value
    betahat = initBeta2D(XtX,XtY)

    # Get ete
    ete = ssr2D(YtX, YtY, XtX, betahat)

    # Get the initial sigma2 value
    sigma2hat = initSigma22D(ete,n)

    # Residuals
    e = Y - X @ betahat

    # Decide on a random factor
    k = np.random.randint(0,nparams.shape[0])

    # Work out derivative term (niave calculation):
    deriv = np.zeros((nparams[k]*(nparams[k]+1)//2,1))
    for j in np.arange(nlevels[k]):

        # Get indices for factor k level j
        Ikj = faclev_indices2D(k, j, nlevels, nparams)

        # Get Z_(k,j)'ee'Z_(k,j)
        ZkjteetZkj = Z[:,Ikj].transpose() @ e @ e.transpose() @ Z[:,Ikj]

        # Get Z_(k,j)'Z_(k,j)
        ZkjtZkj = Z[:,Ikj].transpose() @ Z[:,Ikj]

        # Work out running sum for derivative
        deriv = deriv + mat2vech2D(1/sigma2hat*ZkjteetZkj - ZkjtZkj)

    # Work out Fisher information matrix (niave calculation)
    fishInfo = np.zeros((nparams[k]*(nparams[k]+1)//2,nparams[k]*(nparams[k]+1)//2))
    iDupMat = invDupMat2D(nparams[k])
    for j in np.arange(nlevels[k]):

        for i in np.arange(nlevels[k]):

            # Get indices for factor k level j
            Ikj = faclev_indices2D(k, j, nlevels, nparams)

            # Get indices for factor k level i
            Iki = faclev_indices2D(k, i, nlevels, nparams)

            # Get Z_(k,i)'Z_(k,j)
            ZkitZkj = Z[:,Iki].transpose() @ Z[:,Ikj]

            fishInfo = fishInfo + iDupMat @ np.kron(ZkitZkj, ZkitZkj) @ iDupMat.transpose()

    # This is the value we are testing against
    vecInitDk_expected = vech2mat2D(np.linalg.inv(fishInfo) @ deriv)

    # Work out the inverse duplication matrices we need.
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):
      
      invDupMatdict[i] = invDupMat2D(nparams[i])

    # Obtain Z'e and e'e
    Zte = ZtY - ZtX @ betahat

    # Now try to obtain the same using the function
    vecInitDk_test = initDk2D(k, ZtZ, Zte, sigma2hat, nlevels, nparams, invDupMatdict)

    # Check if the function gives as expected
    testVal = np.allclose(vecInitDk_test,vecInitDk_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: initDk2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `makeDnnd2D`. It does this by
# checking the function against a predefined example.
#
# =============================================================================
def test_makeDnnd2D():

    # Examples usecase
    test = np.eye(5)
    test[3,3]=-1

    # Expected outcome
    expected = np.eye(5)
    expected[3,3]=0

    # Check if the function gives as expected
    testVal = np.allclose(makeDnnd2D(test),expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: makeDnnd2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `llh2D`. It does this by simulating 
# data and checks that the function behaves the same as niave calculation of 
# the same quantity.
#
# -----------------------------------------------------------------------------
#
# Developers note: Occassionally, with this function, a `det` overflow error is
# given. This occurs because, in this case, the niave calculation we are
# testing against is a particularly bad way of calculating the llh. It is not a 
# flaw with the function `llh2D` though and is safe to ignore.
#
# =============================================================================
def test_llh2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]
    q = np.sum(nlevels*nparams)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Obtain Z'e and e'e
    Zte = ZtY - ZtX @ beta
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Obtain log likelihood for testing
    loglh_test = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D)

    # (Niave) Expected log likelihood
    V = np.eye(n)+(Z @ D @ Z.transpose())
    e = Y - X @ beta
    expected = -0.5*(n*np.log(sigma2)+np.log(np.linalg.det(V))+(1/sigma2)*e.transpose() @ np.linalg.inv(V) @ e)

    # Check if the function gives as expected
    testVal = np.allclose(loglh_test,expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: logllh2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_dldB2D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_get_dldB2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]
    q = np.sum(nlevels*nparams)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Obtain Z'e and X'e
    Zte = ZtY - ZtX @ beta
    Xte = XtY - XtX @ beta

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Get derivative with respect to beta for testing
    test = get_dldB2D(sigma2, Xte, XtZ, DinvIplusZtZD, Zte)

    # Get expected result via (niave) direct computation
    V = np.eye(n) + Z @ D @ Z.transpose()
    expected = (sigma2)**(-1)*(X.transpose() @ np.linalg.inv(V) @ (Y - X @ beta))

    # Check if the function gives as expected
    testVal = np.allclose(test,expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dldB2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_dldsigma22D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_get_dldsigma22D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]
    q = np.sum(nlevels*nparams)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Obtain Z'e and X'e
    Zte = ZtY - ZtX @ beta
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Get derivative with respect to sigma2 for testing
    test = get_dldsigma22D(n, ete, Zte, sigma2, DinvIplusZtZD)

    # Get expected result via (niave) direct computation
    V = np.eye(n) + Z @ D @ Z.transpose()
    e = Y - (X @ beta)
    expected = -n/(2*sigma2) + 1/(2*sigma2**2)*e.transpose() @ np.linalg.inv(V) @ e
    
    # Check if the function gives as expected
    testVal = np.allclose(test,expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dldsigma22D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_dldDk2D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_get_dldDk2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]
    q = np.sum(nlevels*nparams)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Obtain Z'e
    Zte = ZtY - ZtX @ beta

    # Decide on a random factor
    k = np.random.randint(0,nparams.shape[0])

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    
    # Obtain dldDk
    dldDk_test = get_dldDk2D(k, nlevels, nparams, ZtZ, Zte, sigma2, DinvIplusZtZD)

    # (Niave) calculation of dl/dDk
    sqrtinvIplusZDZt = forceSym2D(scipy.linalg.sqrtm(np.eye(n) - Z @ DinvIplusZtZD @ Z.transpose()))
    for j in np.arange(nlevels[k]):

        # Indices for factor and level
        Ikj = faclev_indices2D(k, j, nlevels, nparams)

        # Work out Z_(k,j)'
        Z_kjt = Z[:,Ikj].transpose()

        # Work out T_(k,j)=Z_(k,j) @ sqrt((I+ZDZ')^(-1))
        Tkj = Z_kjt @ sqrtinvIplusZDZt

        # Work out u = sigma^(-2)*sqrt((I+ZDZ')^(-1))*(Y-X\beta)
        u = 1/np.sqrt(sigma2)*sqrtinvIplusZDZt @ (Y - X @ beta)

        # Work out T_(k,j)u
        Tkju = Tkj @ u

        # Work out T_(k,j)uu'T_(k,j)'
        TkjuTkjut = Tkju @ Tkju.transpose()

        # Work out T_(k,j)T_(k,j)'
        TkjTkjt = Tkj @ Tkj.transpose()

        # Work out the sum of T_(k,j)uu'T_(k,j)' and T_(k,j)T_(k,j)' 
        if j == 0:

            sum1 = TkjuTkjut
            sum2 = TkjTkjt

        else:

            sum1 = sum1 + TkjuTkjut
            sum2 = sum2 + TkjTkjt
        
    # Work out expected derivative.
    dldDk_expected = 0.5*(sum1-sum2)

    # Check if the function gives as expected
    testVal = np.allclose(dldDk_test,dldDk_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dldDk2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_covdldbeta2D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_get_covdldbeta2D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]
    q = np.sum(nlevels*nparams)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Obtain Z'e
    Zte = ZtY - ZtX @ beta

    # Calculate covariance of dl/dbeta (niavely)
    V = np.eye(n) + Z @ D @ Z.transpose()
    covdldbeta_expected = sigma2**(-1) * (X.transpose() @ np.linalg.inv(V) @ X)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Get the covariance of dl/dbeta from the function.
    covdldbeta_test = get_covdldbeta2D(XtZ, XtX, ZtZ, DinvIplusZtZD, sigma2)

    # Check if the function gives as expected
    testVal = np.allclose(covdldbeta_test,covdldbeta_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_covdldbeta2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_covdldDksigma22D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_get_covdldDkdsigma22D():

    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]
    q = np.sum(nlevels*nparams)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Work out the inverse duplication matrices we need.
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):
      
      invDupMatdict[i] = invDupMat2D(nparams[i])

    # Decide on a random factor
    k = np.random.randint(0,nparams.shape[0])

    # (Niave) computation of the covariance 
    sqrtinvIplusZDZt = forceSym2D(scipy.linalg.sqrtm(np.eye(n) - Z @ DinvIplusZtZD @ Z.transpose()))
    for j in np.arange(nlevels[k]):

        # Indices for factor and level
        Ikj = faclev_indices2D(k, j, nlevels, nparams)

        # Work out Z_(k,j)'
        Z_kjt = Z[:,Ikj].transpose()

        # Work out T_(k,j)=Z_(k,j) @ sqrt((I+ZDZ')^(-1))
        Tkj = Z_kjt @ sqrtinvIplusZDZt

        # Work out T_(k,j)T_(k,j)'
        TkjTkjt = Tkj @ Tkj.transpose()

        # Work out the sum of T_(k,j)T_(k,j)' 
        if j == 0:

            sumTTt = TkjTkjt

        else:

            sumTTt = sumTTt + TkjTkjt

    # Final niave computation
    covdldDkdsigma2_expected = 1/(2*sigma2)*(invDupMatdict[k] @ mat2vec2D(sumTTt))

    # Computation using the function
    covdldDkdsigma2_test = get_covdldDkdsigma22D(k, sigma2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict, vec=False)
  
    # Check if the function gives as expected
    testVal = np.allclose(covdldDkdsigma2_test,covdldDkdsigma2_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_covdldDkdsigma22D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_covdldDk1Dk22D`. It does this by 
# simulating data and checks that the function behaves the same as niave
# calculation of the same quantity.
#
# =============================================================================
def test_get_covdldDk1Dk22D():
    
    # Generate some test data
    Y,X,Z,nlevels,nparams,beta,sigma2,b,D = genTestData()
    n = X.shape[0]
    q = np.sum(nlevels*nparams)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats(Y,Z,X)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Obtain (I+ZDZ')^(-1/2), for niave computation
    IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
    invIplusZDZt = np.linalg.inv(IplusZDZt)
    invhalfIplusZDZt = scipy.linalg.sqrtm(np.linalg.inv(IplusZDZt))

    # Decide on a random factor
    k1 = np.random.randint(0,nparams.shape[0])

    # Decide on another random factor
    k2 = np.random.randint(0,nparams.shape[0])

    # Work out the inverse duplication matrices we need.
    invDupMatdict = dict()
    for i in np.arange(len(nparams)):
      
      invDupMatdict[i] = invDupMat2D(nparams[i])

    # Test the function
    covdldDk1Dk2_test = get_covdldDk1Dk22D(k1, k2, nlevels, nparams, ZtZ, DinvIplusZtZD, invDupMatdict)

    # Perform the same calculation niavely
    for j in np.arange(nlevels[k1]):

        # Obtain indices for factor k level j
        Ikj = faclev_indices2D(k1, j, nlevels, nparams)

        # Work out T_(k,j) = Z_(k,j)(I+ZDZ')^(-1/2)
        Tkj = Z[:,Ikj].transpose() @ invhalfIplusZDZt 

        # Sum T_(k,j) kron T_(k,j)
        if j == 0:

            sumTkT = np.kron(Tkj,Tkj)

        else:

            sumTkT = np.kron(Tkj,Tkj) + sumTkT

    # Do the same again but for the second factor
    for j in np.arange(nlevels[k2]):

        # Obtain indices for factor k level j
        Ikj = faclev_indices2D(k2, j, nlevels, nparams)

        # Work out T_(k,j) = Z_(k,j)(I+ZDZ')^(-1/2)
        Tkj = Z[:,Ikj].transpose() @ invhalfIplusZDZt 

        # Transpose T_(k,j)
        Tkjt = Tkj.transpose()

        # Sum T_(k,j)' kron T_(k,j)'
        if j == 0:

            sumTtkTt = np.kron(Tkjt,Tkjt)

        else:

            sumTtkTt = np.kron(Tkjt,Tkjt) + sumTtkTt

    # Expected result from the function
    covdldDk1Dk2_expected = 1/2 * invDupMatdict[k1] @ sumTkT @ sumTtkTt @ invDupMatdict[k2].transpose()

    # Check if the function gives as expected
    testVal = np.allclose(covdldDk1Dk2_test,covdldDk1Dk2_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_covdldDk1Dk22D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)



# =============================================================================
#
# The below function tests the function `mapping2D`. It does this by checking
# the function against a predefined example.
#
# =============================================================================
def test_mapping2D():

    # Theta values to put in the array
    theta = np.array([1,2,3,4])

    # Theta indices
    theta_inds = np.array([0,1,2,0,1,2,0,1,2,3,3,3])

    # Row and column indices
    r_inds = np.array([0,1,1,2,3,3,4,5,5,6,7,8])
    c_inds = np.array([0,0,1,2,2,3,4,4,5,6,7,8])

    # Matrix we expect
    expected = np.array([[1,0,0,0,0,0,0,0,0],
                         [2,3,0,0,0,0,0,0,0],
                         [0,0,1,0,0,0,0,0,0],
                         [0,0,2,3,0,0,0,0,0],
                         [0,0,0,0,1,0,0,0,0],
                         [0,0,0,0,2,3,0,0,0],
                         [0,0,0,0,0,0,4,0,0],
                         [0,0,0,0,0,0,0,4,0],
                         [0,0,0,0,0,0,0,0,4]])

    # Test result
    test = np.array(cvxopt.matrix(mapping2D(theta, theta_inds, r_inds, c_inds)))

    # Check if the function gives as expected
    testVal = np.allclose(test,expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mapping2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `sparse_chol2D`. It does this by 
# generating random matrices and checking the output has the correct
# properties.
#
# =============================================================================
def test_sparse_chol2D():

    # We need to set the cholmod options to 2
    cholmod.options['supernodal']=2

    # Random matrix dimensions
    n = np.random.randint(5,20)

    # Random sparse positive definite matrix
    M = np.random.randn(n,n)
    M = M @ M.transpose() + np.eye(n)*0.05
    M[M<0]=0

    # Convert M to cvxopt
    M = cvxopt.sparse(cvxopt.matrix(M))
    
    # Obtain cholesky factorization
    cholObject = sparse_chol2D(M, perm=None, retF=False, retP=True, retL=True)

    # Lower cholesky
    L = cholObject['L']

    # Permutation
    P = cholObject['P']

    # Estimated M given sparse cholesky
    M_est = L*L.trans()
    M_est = np.array(cvxopt.matrix(M_est))

    # Permuted M 
    M_permuted = np.array(cvxopt.matrix(M[P,P]))

    # Check if the function give a valid sparse cholesky decomposition
    testVal = np.allclose(M_est,M_permuted)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: sparse_chol2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `get_mapping2D`. It does this by
# checking the function against a predefined example.
#
# =============================================================================
def test_get_mapping2D():

    # Example nlevels and nparams
    nlevels = np.array([3,3])
    nparams = np.array([2,1])

    # Expected theta indices
    t_inds_expected = np.array([0,1,2,0,1,2,0,1,2,3,3,3])

    # Expected row and column indices
    r_inds_expected = np.array([0,1,1,2,3,3,4,5,5,6,7,8])
    c_inds_expected = np.array([0,0,1,2,2,3,4,4,5,6,7,8])

    # Get the functions output
    t_inds_test, r_inds_test, c_inds_test =get_mapping2D(nlevels, nparams)

    # Check if the function gives as expected
    testVal = np.allclose(t_inds_test,t_inds_expected) and np.allclose(r_inds_test,r_inds_expected) and np.allclose(c_inds_test,c_inds_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_mapping2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function runs all unit tests and outputs the results.
#
# =============================================================================
def run_all2D():

    # Record passed and failed tests.
    passedTests = np.array([])
    failedTests = np.array([])


    # Test mat2vec2D
    name = 'mat2vec2D'
    result = test_mat2vec2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test mat2vech2D
    name = 'mat2vech2D'
    result = test_mat2vech2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test vec2mat2D
    name = 'vec2mat2D'
    result = test_vec2mat2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test vech2mat2D
    name = 'vech2mat2D'
    result = test_vech2mat2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test vec2vech2D
    name = 'vec2vech2D'
    result = test_vec2vech2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test blockInverse2D
    name = 'blockInverse2D'
    result = test_blockInverse2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test recursiveInverse2D
    name = 'recursiveInverse2D'
    result = test_recursiveInverse2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test dupMat2D
    name = 'dupMat2D'
    result = test_dupMat2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test invDupMat2D
    name = 'invDupMat2D'
    result = test_invDupMat2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test comMat2D
    name = 'comMat2D'
    result = test_comMat2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test permOfIkKkI2D
    name = 'permOfIkKkI2D'
    result = test_permOfIkKkI2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test block2stacked2D
    name = 'block2stacked2D'
    result = test_block2stacked2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test mat2vecb2D
    name = 'mat2vecb2D'
    result = test_mat2vecb2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test sumAijBijt2D
    name = 'sumAijBijt2D'
    result = test_sumAijBijt2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test sumAijKronBij2D
    name = 'sumAijKronBij2D'
    result = test_sumAijKronBij2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test forceSym2D
    name = 'forceSym2D'
    result = test_forceSym2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test ssr2D
    name = 'ssr2D'
    result = test_ssr2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test fac_indices2D
    name = 'fac_indices2D'
    result = test_fac_indices2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test faclev_indices2D
    name = 'faclev_indices2D'
    result = test_faclev_indices2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test initBeta2D
    name = 'initBeta2D'
    result = test_initBeta2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test initSigma22D
    name = 'initSigma22D'
    result = test_initSigma22D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test initDk2D
    name = 'initDk2D'
    result = test_initDk2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test makeDnnd2D
    name = 'makeDnnd2D'
    result = test_makeDnnd2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test llh2D
    name = 'llh2D'
    result = test_llh2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_dldB2D
    name = 'get_dldB2D'
    result = test_get_dldB2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)

    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_dldsigma22D
    name = 'get_dldsigma22D'
    result = test_get_dldsigma22D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_dldDk2D
    name = 'get_dldDk2D'
    result = test_get_dldDk2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_covdldbeta2D
    name = 'get_covdldbeta2D'
    result = test_get_covdldbeta2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_covdldDkdsigma22D
    name = 'get_covdldDkdsigma22D'
    result = test_get_covdldDkdsigma22D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_covdldDk1Dk22D
    name = 'get_covdldDk1Dk22D'
    result = test_get_covdldDk1Dk22D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test mapping2D
    name = 'mapping2D'
    result = test_mapping2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test sparse_chol2D
    name = 'sparse_chol2D'
    result = test_sparse_chol2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_mapping2D
    name = 'get_mapping2D'
    result = test_get_mapping2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    print('=============================================================')

    print('Tests completed')
    print('-------------------------------------------------------------')
    print('Summary:')
    print('Passed Tests: ', passedTests)
    print('Failed Tests: ', failedTests)
