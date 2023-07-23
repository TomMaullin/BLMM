import numpy as np
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack

# ============================================================================
#
# The below function applies a mapping to a vector of parameters. (Used in 
# PeLS - equivalent of what is described in Bates 2015)
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
# - `nraneffs`: A vector containing the number of random effects for each
#              factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#              random effects and the second factor has 1 random effect.
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
def get_mapping2D(nlevels, nraneffs):

    # Work out how many factors there are
    n_f = len(nlevels)

    # Quick check that nlevels and nraneffs are the same length
    if len(nlevels)!=len(nraneffs):
        raise Exception('The number of random effects and number of levels should be recorded for every grouping factor.')

    # Work out how many lambda components needed for each factor
    n_lamcomps = (np.multiply(nraneffs,(nraneffs+1))/2).astype(np.int64)

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
        row_inds_tri, col_inds_tri = np.tril_indices(nraneffs[i])

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
            block_index = block_index + nraneffs[i]

    # Create lambda as a sparse matrix
    #lambda_theta = spmatrix(theta_repeated.tolist(), row_indices.astype(np.int64), col_indices.astype(np.int64))

    # Return lambda
    return(theta_repeated_inds, row_indices, col_indices)