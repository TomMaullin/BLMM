import os
import sys
import numpy as np
import time
import scipy.sparse
from scipy import stats

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from lib.npMatrix2d import *
from genTestDat import prodMats2D, genTestData2D

# =============================================================================
# This file contains all unit tests for the functions given in the 
# npMatrix2D.py file.
#
# Author: Tom Maullin
# Last edited: 21/03/2020
# =============================================================================


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
# The below function tests the function `mat2vechTri2D`. It does this by
# checking the function against a predefined example.
#
# =============================================================================
def test_mat2vechTri2D():

    # Test vector
    expected = (np.arange(28)+1).reshape(28,1)

    # Empty n by n matrix.
    mat = np.array([[1,0,0,0,0,0,0],
                    [2,8,0,0,0,0,0],
                    [3,9,14,0,0,0,0],
                    [4,10,15,19,0,0,0],
                    [5,11,16,20,23,0,0],
                    [6,12,17,21,24,26,0],
                    [7,13,18,22,25,27,28]])

    # Test output
    test = mat2vechTri2D(mat)
    
    # Test
    testVal = np.allclose(test, expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: mat2vechTri2D')
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
# The below function tests the function `vechTri2mat2D`. It does this by
# checking the function against a predefined example.
#
# =============================================================================
def test_vechTri2mat2D():

    # Test vector
    vech=np.arange(28)+1

    # Empty n by n matrix.
    expected = np.array([[1,0,0,0,0,0,0],
                         [2,8,0,0,0,0,0],
                         [3,9,14,0,0,0,0],
                         [4,10,15,19,0,0,0],
                         [5,11,16,20,23,0,0],
                         [6,12,17,21,24,26,0],
                         [7,13,18,22,25,27,28]])

    # Test output
    test = vechTri2mat2D(vech)
    
    # Test
    testVal = np.allclose(test, expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: vechTri2mat2D')
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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()

    # Work out q
    q = np.sum(nlevels*nraneffs)

    # Work out I+Z'ZD
    ZtZ = Z.transpose() @ Z 

    # Sparse version
    ZtZ_sp = scipy.sparse.csr_matrix(ZtZ)

    # Recursive inverse
    ZtZinv_test = recursiveInverse2D(ZtZ_sp, nraneffs, nlevels)

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
# The below function tests the function `elimMat2D`. It does this by checking 
# the elimination matrix given has the desired effect on a vector.
#
# =============================================================================
def test_elimMat2D():

    # Choose dimension for random matrix
    m = np.random.randint(3,14)

    # Make random lower triangular matrix
    a = np.tril(np.random.randn(m**2).reshape(m,m))

    # Convert to vec
    avec = mat2vec2D(a)

    # Expected result
    expected = avec[avec!=0].reshape(avec[avec!=0].shape[0],1)

    # Result given using elimMatFunction
    test = elimMat2D(m) @ avec

    # Test
    testVal = np.allclose(test, expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: elimMat2D')
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
    l1 = np.random.randint(1,300)
    l2 = np.random.randint(1,300)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

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

    # Test nlevels, nraneffs,k
    k = 2
    nlevels = np.array([3,4,2,8])
    nraneffs = np.array([1,2,3,4])

    # Expected output
    expected = np.arange(11,17)

    # Check if the function gives as expected
    testVal = np.allclose(fac_indices2D(k,nlevels,nraneffs),expected)

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

    # Test nlevels, nraneffs,k
    k = 2
    j = 1
    nlevels = np.array([3,4,2,8])
    nraneffs = np.array([1,2,3,4])

    # Expected output
    expected = np.arange(14,17)

    # Check if the function gives as expected
    testVal = np.allclose(faclev_indices2D(k,j,nlevels,nraneffs),expected)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()

    # Calculate OLS estimate using test data.
    expected = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Get the initial beta value
    betahat = initBeta2D(XtX,XtY)

    # Get ete
    ete = ssr2D(YtX, YtY, XtX, betahat)

    # Get the initial sigma2 value
    sigma2hat = initSigma22D(ete,n)

    # Residuals
    e = Y - X @ betahat

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # Work out derivative term (niave calculation):
    deriv = np.zeros((nraneffs[k]*(nraneffs[k]+1)//2,1))
    dupMatT = dupMat2D(nraneffs[k]).transpose()
    for j in np.arange(nlevels[k]):

        # Get indices for factor k level j
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

        # Get Z_(k,j)'ee'Z_(k,j)
        ZkjteetZkj = Z[:,Ikj].transpose() @ e @ e.transpose() @ Z[:,Ikj]

        # Get Z_(k,j)'Z_(k,j)
        ZkjtZkj = Z[:,Ikj].transpose() @ Z[:,Ikj]

        # Work out running sum for derivative
        deriv = deriv + dupMatT @ mat2vec2D(1/sigma2hat*ZkjteetZkj - ZkjtZkj)

    # Work out Fisher information matrix (niave calculation)
    fishInfo = np.zeros((nraneffs[k]*(nraneffs[k]+1)//2,nraneffs[k]*(nraneffs[k]+1)//2))
    for j in np.arange(nlevels[k]):

        for i in np.arange(nlevels[k]):

            # Get indices for factor k level j
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Get indices for factor k level i
            Iki = faclev_indices2D(k, i, nlevels, nraneffs)

            # Get Z_(k,i)'Z_(k,j)
            ZkitZkj = Z[:,Iki].transpose() @ Z[:,Ikj]

            fishInfo = fishInfo + dupMatT @ np.kron(ZkitZkj, ZkitZkj) @ dupMatT.transpose()

    # This is the value we are testing against
    vecInitDk_expected = vech2mat2D(np.linalg.inv(fishInfo) @ deriv)

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()

    # Obtain Z'e and e'e
    Zte = ZtY - ZtX @ betahat

    # Now try to obtain the same using the function
    vecInitDk_test = initDk2D(k, ZtZ, Zte, sigma2hat, nlevels, nraneffs, dupMatTdict)

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
# The below function tests the function `makeDpd2D`. It does this by checking
# the function against a predefined example.
#
# =============================================================================
def test_makeDpd2D():

    # Examples usecase
    test = np.eye(5)
    test[3,3]=-1

    # Expected outcome
    expected = np.eye(5)
    expected[3,3]=1e-6

    # Check if the function gives as expected
    testVal = np.allclose(makeDpd2D(test),expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: makeDpd2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)

    return(result)


# =============================================================================
#
# The below function tests the function `llh2D`. It does this by simulating 
# data and checks that the function behaves the same as niave calculation of 
# the same quantity.
#
# =============================================================================
def test_llh2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    q = np.sum(nlevels*nraneffs)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

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
    expected = -0.5*(n*np.log(sigma2)+np.linalg.slogdet(V)[0]*np.linalg.slogdet(V)[1]+(1/sigma2)*e.transpose() @ np.linalg.inv(V) @ e)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    q = np.sum(nlevels*nraneffs)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    q = np.sum(nlevels*nraneffs)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    q = np.sum(nlevels*nraneffs)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Obtain Z'e
    Zte = ZtY - ZtX @ beta

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    
    # Obtain dldDk
    dldDk_test,ZtZmat = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD)
    dldDk_test,_ = get_dldDk2D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat)

    # (Niave) calculation of dl/dDk
    sqrtinvIplusZDZt = forceSym2D(scipy.linalg.sqrtm(np.eye(n) - Z @ DinvIplusZtZD @ Z.transpose()))
    for j in np.arange(nlevels[k]):

        # Indices for factor and level
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    q = np.sum(nlevels*nraneffs)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    q = np.sum(nlevels*nraneffs)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()

    # Decide on a random factor
    k = np.random.randint(0,nraneffs.shape[0])

    # (Niave) computation of the covariance 
    sqrtinvIplusZDZt = forceSym2D(scipy.linalg.sqrtm(np.eye(n) - Z @ DinvIplusZtZD @ Z.transpose()))
    for j in np.arange(nlevels[k]):

        # Indices for factor and level
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

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
    covdldDkdsigma2_expected = 1/(2*sigma2)*(dupMatTdict[k] @ mat2vec2D(sumTTt))

    # Computation using the function
    covdldDkdsigma2_test = get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=False,ZtZmat=None)[0]
  
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
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    q = np.sum(nlevels*nraneffs)

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Obtain (I+ZDZ')^(-1/2), for niave computation
    IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
    invIplusZDZt = np.linalg.inv(IplusZDZt)
    invhalfIplusZDZt = scipy.linalg.sqrtm(np.linalg.inv(IplusZDZt))

    # Decide on a random factor
    k1 = np.random.randint(0,nraneffs.shape[0])

    # Decide on another random factor
    k2 = np.random.randint(0,nraneffs.shape[0])

    # Work out the transpose duplication matrices we need.
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):
      
      dupMatTdict[i] = dupMat2D(nraneffs[i]).transpose()

    # Test the function
    covdldDk1Dk2_test,perm = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)
    covdldDk1Dk2_test,_ = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict,perm=perm)

    # Perform the same calculation niavely
    for j in np.arange(nlevels[k1]):

        # Obtain indices for factor k level j
        Ikj = faclev_indices2D(k1, j, nlevels, nraneffs)

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
        Ikj = faclev_indices2D(k2, j, nlevels, nraneffs)

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
    covdldDk1Dk2_expected = 1/2 * dupMatTdict[k1] @ sumTkT @ sumTtkTt @ dupMatTdict[k2].transpose()

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
# The below function tests the function `get_resms2D`. It does this by simulating
# random test data and testing against niave computation using `ssr2D`.
#
# =============================================================================
def test_get_resms2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # First test spatially varying
    resms_test = get_resms2D(YtX, YtY, XtX, beta, n, p)
    resms_expected = ssr2D(YtX, YtY, XtX, beta)/(n-p)

    # Check if results are all close.
    testVal = np.allclose(resms_test,resms_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_resms2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)
    

# =============================================================================
#
# The below function tests the function `covB2D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_covB2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB_expected = np.linalg.inv(X.transpose() @ invV @ X/sigma2)

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    covB_test = get_covB2D(XtX, XtZ, DinvIplusZtZD, sigma2)

    # Check if results are all close.
    testVal = np.allclose(covB_test,covB_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: covB2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `varLB2D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_varLB2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB = np.linalg.inv(X.transpose() @ invV @ X/sigma2)
    
    # Variance of L times beta
    varLB_expected = L @ covB @ L.transpose()

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    varLB_test = get_varLB2D(L, XtX, XtZ, DinvIplusZtZD, sigma2)

    # Check if results are all close.
    testVal = np.allclose(varLB_test,varLB_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: varLB2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_T2D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_get_T2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB = np.linalg.inv(X.transpose() @ invV @ X/sigma2)
    
    # Variance of L times beta
    varLB = L @ covB @ L.transpose()

    # Work out T
    T_expected = LB/np.sqrt(varLB) 

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    T_test = get_T2D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2)

    # Check if results are all close.
    testVal = np.allclose(T_test,T_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_T2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_F2D`. It does this by simulating
# random test data and testing against niave calculation.
#
# =============================================================================
def test_get_F2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Test contrast vector
    L = np.eye(p)
    L[:,0]=1

    # Expected, niave calculation.
    # ----------------------------------------------------------------------

    # rank of L
    rL = np.linalg.matrix_rank(L)

    # L times beta
    LB = L @ beta

    # V^(-1)
    IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    covB = np.linalg.inv(X.transpose() @ invV @ X/sigma2)
    
    # Variance of L times beta
    varLB = L @ covB @ L.transpose()

    # Work out F
    F_expected = LB.transpose() @ np.linalg.inv(varLB) @ LB/rL

    # Using function
    # ----------------------------------------------------------------------

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Run F test
    F_test = get_F2D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2)

    # Check if results are all close.
    testVal = np.allclose(F_test,F_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_F2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_R22D`. It does this by simulating
# a random example and testing against niave calculation.
#
# =============================================================================
def test_get_R22D():

    # Random "F" statistic
    F = np.random.randn(1)[0]

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,np.random.randint(2,10)))
    L[0,0]=1

    # Rank of contrast vector
    rL = np.linalg.matrix_rank(L)

    # Random degrees of freedom
    df_denom = np.random.binomial(1,0.7)+1

    # Expected R^2
    R2_expected = (rL*F)/(rL*F + df_denom)

    # Test R^2
    R2_test = get_R22D(L, F, df_denom)

    # Check if results are all close.
    testVal = np.allclose(R2_test,R2_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_R22D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `T2P2D`. It does this by simulating
# a random example and testing against niave calculation.
#
# =============================================================================
def test_T2P2D():

    # Random "T" statistic
    T = np.random.randn(1)[0]

    # Random minlog value
    minlog = -np.random.randint(500,1000)

    # Random degrees of freedom
    df = np.random.binomial(1,0.7)+1

    # Expected P value
    P_expected = -np.log10(1-stats.t.cdf(T, df))

    # Remove infs
    if np.isinf(P_expected) and P_expected<0:

        P_expected = minlog

    P_test = T2P2D(T,df,minlog)

    # Check if results are all close.
    testVal = np.allclose(P_test,P_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: T2P2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `F2P2D`. It does this by simulating
# a random example and testing against niave calculation.
#
# =============================================================================
def test_F2P2D():

    # Random "F" statistics
    F = np.random.randn(1)**2

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,np.random.randint(2,10)))
    L[0,0]=1

    # Random minlog value
    minlog = -np.random.randint(500,1000)

    # Random degrees of freedom
    df_denom = np.random.binomial(1,0.7)+1

    # Expected P value
    P_expected = -np.log10(1-stats.f.cdf(F, np.linalg.matrix_rank(L), df_denom))

    # Remove infs
    if np.isinf(P_expected) and P_expected<0:

        P_expected = minlog

    P_test = F2P2D(F, L, df_denom, minlog)

    # Check if results are all close.
    testVal = np.allclose(P_test,P_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: F2P2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_swdf_T2D`. It does this by
# simulating random test data and testing against niave calculation.
#
# Note: This test assumes the correctness of the functions `get_InfoMat2D` and
# `get_dS22D`.
#
# =============================================================================
def test_get_swdf_T2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Get D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB2D(L, XtX, XtZ, DinvIplusZtZD, sigma2)
    
    # Get derivative of S^2
    dS2 = get_dS22D(nraneffs, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2)

    # Get Fisher information matrix
    InfoMat = get_InfoMat2D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ)

    # Calculate df estimator
    swdf_expected = 2*(S2**2)/(dS2.transpose() @ np.linalg.inv(InfoMat) @ dS2)

    swdf_test = get_swdf_T2D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs)

    # Check if results are all close.
    testVal = np.allclose(swdf_test,swdf_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_swdf_T2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_swdf_F2D`. It does this by
# simulating random test data and testing against niave calculation.
#
# Note: This test assumes the correctness of the functions `get_swdf_T2D`.
#
# =============================================================================
def test_get_swdf_F2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

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
    swdf_expected = 2*sum_swdf_adj/(sum_swdf_adj-rL)

    # Function version 
    swdf_test = get_swdf_F2D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs)

    # Check if results are all close.
    testVal = np.allclose(swdf_test,swdf_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_swdf_F2D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_dS22D`. It does this by
# simulating random test data and testing against niave calculation.
#
# =============================================================================
def test_get_dS22D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Test contrast vector
    L = np.random.binomial(1,0.5,size=(1,p))
    L[0,0]=1

    # Niave calculation for one voxel.
    # ----------------------------------------------------------------------------------
    IplusZDZt = np.eye(n) + Z @ D @ Z.transpose()
    invV = np.linalg.inv(IplusZDZt)

    # Calculate X'V^{-1}X
    XtiVX = X.transpose() @ invV @ X 

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2_expected = np.zeros((1+np.int32(np.sum(nraneffs*(nraneffs+1)/2)),1))

    # Work out indices for each start of each component of vector 
    # i.e. [dS2/dsigm2, dS2/vechD1,...dS2/vechDr]
    DerivInds = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + 1)
    DerivInds = np.insert(DerivInds,0,1)

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.inv(XtiVX) @ L.transpose()

    # Add to dS2
    dS2_expected[0:1] = dS2dsigma2.reshape(dS2_expected[0:1].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((np.int32(nraneffs[k]*(nraneffs[k]+1)/2),1))

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)
                    
            # Work out Z_(k,j)'
            Zkjt = Z[:,Ikj].transpose()

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = Zkjt @ invV @ X

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.inv(XtiVX) @ L.transpose()
            
            # Duplication matrix transposed
            dupMatT = dupMat2D(nraneffs[k]).transpose()
            
            # Sum terms
            dS2dvechDk = dS2dvechDk + dupMatT @ mat2vec2D(np.kron(K,K.transpose()))

        # Multiply by sigma^2
        dS2dvechDk = sigma2*dS2dvechDk

        # Add to dS2
        dS2_expected[DerivInds[k]:DerivInds[k+1]] = dS2dvechDk.reshape(dS2_expected[DerivInds[k]:DerivInds[k+1]].shape)


    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Obtain result from function
    dS2_test = get_dS22D(nraneffs, nlevels, L, XtX, XtZ, ZtZ, DinvIplusZtZD, sigma2)

    # Check if results are all close.
    testVal = np.allclose(dS2_test,dS2_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_dS22D')
    print('-------------------------------------------------------------')
    print('Result: ', result)
    
    return(result)


# =============================================================================
#
# The below function tests the function `get_InfoMat2D`. It does this by
# simulating random test data and testing against niave calculation.
#
# =============================================================================
def test_get_InfoMat2D():

    # Generate some test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D = genTestData2D()
    q = np.sum(np.dot(nraneffs,nlevels))
    n = X.shape[0]
    p = X.shape[1]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # Obtain D(I+Z'ZD)^(-1)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

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
    FI_expected = np.zeros((tnp,tnp))
    
    # Covariance of dl/dsigma2
    covdldsigma2 = n/(2*(sigma2**2))
    
    # Add dl/dsigma2 covariance
    FI_expected[0,0] = covdldsigma2

    # Add dl/dsigma2 dl/dD covariance
    for k in np.arange(len(nraneffs)):

        # Get covariance of dldsigma and dldD      
        covdldsigmadD = get_covdldDkdsigma22D(k, sigma2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)[0].reshape(FishIndsDk[k+1]-FishIndsDk[k])

        # Assign to the relevant block
        FI_expected[0, FishIndsDk[k]:FishIndsDk[k+1]] = covdldsigmadD
        FI_expected[FishIndsDk[k]:FishIndsDk[k+1],0:1] = FI_expected[0:1, FishIndsDk[k]:FishIndsDk[k+1]].transpose()
      
    # Add dl/dD covariance
    for k1 in np.arange(len(nraneffs)):

        for k2 in np.arange(k1+1):

            IndsDk1 = np.arange(FishIndsDk[k1],FishIndsDk[k1+1])
            IndsDk2 = np.arange(FishIndsDk[k2],FishIndsDk[k2+1])

            # Get covariance between D_k1 and D_k2 
            covdldDk1dDk2 = get_covdldDk1Dk22D(k1, k2, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict)[0]

            # Add to FImat
            FI_expected[np.ix_(IndsDk1, IndsDk2)] = covdldDk1dDk2
            FI_expected[np.ix_(IndsDk2, IndsDk1)] = FI_expected[np.ix_(IndsDk1, IndsDk2)].transpose()

    FI_test = get_InfoMat2D(DinvIplusZtZD, sigma2, n, nlevels, nraneffs, ZtZ)

    # Check if results are all close.
    testVal = np.allclose(FI_test,FI_expected)

    # Result
    if testVal:
        result = 'Passed'
    else:
        result = 'Failed'

    print('=============================================================')
    print('Unit test for: get_InfoMat2D')
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


    # Test mat2vechTri2D
    name = 'mat2vechTri2D'
    result = test_mat2vechTri2D()
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


    # Test vechTri2mat2D
    name = 'vechTri2mat2D'
    result = test_vechTri2mat2D()
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


    # Test elimMat2D
    name = 'elimMat2D'
    result = test_elimMat2D()
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


    # Test makeDpd2D
    name = 'makeDpd2D'
    result = test_makeDpd2D()
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


    # Test get_resms2D
    name = 'get_resms2D'
    result = test_get_resms2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)

        
    # Test covB2D
    name = 'covB2D'
    result = test_covB2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test varLB2D
    name = 'varLB2D'
    result = test_varLB2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_T2D
    name = 'get_T2D'
    result = test_get_T2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_F2D
    name = 'get_F2D'
    result = test_get_F2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_R22D
    name = 'get_R22D'
    result = test_get_R22D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test T2P2D
    name = 'T2P2D'
    result = test_T2P2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test F2P2D
    name = 'F2P2D'
    result = test_F2P2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_swdf_T2D
    name = 'get_swdf_T2D'
    result = test_get_swdf_T2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_swdf_F2D
    name = 'get_swdf_F2D'
    result = test_get_swdf_F2D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_dS22D
    name = 'get_dS22D'
    result = test_get_dS22D()
    # Add result to arrays.
    if result=='Passed':
        passedTests = np.append(passedTests, name)
    if result=='Failed':
        failedTests = np.append(failedTests, name)


    # Test get_InfoMat2D
    name = 'get_InfoMat2D'
    result = test_get_InfoMat2D()
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
