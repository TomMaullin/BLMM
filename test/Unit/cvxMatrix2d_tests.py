import numpy as np
import sys
import os
import cvxopt
from cvxopt import cholmod

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from blmm.src.cvxMatrix2d import *

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

    # Example nlevels and nraneffs
    nlevels = np.array([3,3])
    nraneffs = np.array([2,1])

    # Expected theta indices
    t_inds_expected = np.array([0,1,2,0,1,2,0,1,2,3,3,3])

    # Expected row and column indices
    r_inds_expected = np.array([0,1,1,2,3,3,4,5,5,6,7,8])
    c_inds_expected = np.array([0,0,1,2,2,3,4,4,5,6,7,8])

    # Get the functions output
    t_inds_test, r_inds_test, c_inds_test =get_mapping2D(nlevels, nraneffs)

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
