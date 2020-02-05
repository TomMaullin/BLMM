import numpy as np
import scipy.sparse
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack
from lib.tools2d import faclev_indices2D
import numdifftools as nd


# ============================================================================
#
# The function below is a wrapper function for all Sattherthwaite degrees of 
# freedom estimation used by BLMM.
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
def SattherthwaiteDoF(statType,estType,L,XtX,XtY,XtZ,YtX,YtY,YtZ):

	pass

def SW_lmerTest(theta3D,tinds,rinds,cinds):

	# Get the sigma^2 and D estimates.
	for i in np.arange(theta3D.shape[0]):

		# Get current theta
        theta = theta3D[i,:]

        # Convert product matrices to CVXopt form
        XtY_current = cvxopt.matrix(XtY[i,:,:])
        YtX_current = cvxopt.matrix(YtX[i,:,:])
        YtY_current = cvxopt.matrix(YtY[i,:,:])
        YtZ_current = cvxopt.matrix(YtZ[i,:,:])
        ZtY_current = cvxopt.matrix(ZtY[i,:,:])

        # Obtain beta estimate
        beta = np.array(PLS2D_getBeta(theta, ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, YtZ_current, XtZ_current, YtY_current, n, P, tinds, rinds, cinds))

        # Obtain sigma^2 estimate
        sigma2 = PLS2D_getSigma2(theta, ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, YtZ_current, XtZ_current, YtY_current, n, P, I, tinds, rinds, cinds)
        
        # Obtain D estimate
        D = np.array(matrix(PLS2D_getD(theta, tinds, rinds, cinds, sigma2)))


        #NTS CURRENTLY FOR SPARSE CHOL, NOT (\sigma,SPCHOL(D))
        #ALSO MIGHT HAVE PROBLEMS WITH CVXOPT CONVERSION

        # How to get the log likelihood
        def llhPLS(t, ZtX=ZtX_current, ZtY=ZtY_current, XtX=XtX_current, ZtZ=ZtZ_current, XtY=XtY_current, 
        		   YtX=YtX_current, YtZ=YtZ_current, XtZ=XtZ_current, YtY=YtY_current, n=n, P=P, I=I, 
        		   tinds=tinds, rinds=rinds, cinds=cinds): 
        	return PLS2D(t, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds)

        # Estimate hessian
        H = nd.Hessian(llhPLS)(theta)

        # Estimate Jacobian




### NTS CHANGE THETA TO GAMMA TO SHOW DIFF BETWEEN LMER AND LMERTEST THETA
def gamma2theta():

	pass

def theta2gamma():

	pass


def S2():#TODO inputs

	# Get the mapping from the lower cholesky decomposition to the full D matrix.
	#tinds,rinds,cinds=get_mapping2D(nlevels, nparams)

	# Calculate X'V^{-1}X=X'(I+ZDZ')^{-1}X=X'X-X'Z(I+DZ'Z)^{-1}DZ'X
	XtiVX = XtX - XtZ @ np.linalg.inv(I + D @ ZtZ) @ D @ ZtX

	# Calculate S^2 = sigma^2L(X'V^{-1}X)L'
	S2 = sigma2*L @ np.linalg.inv(XtiVX) @ L.transpose()

	return(S2)


def dS2dgamma(): # TODO inputs

	# Calculate X'V^{-1}X=X'(I+ZDZ')^{-1}X=X'X-X'Z(I+DZ'Z)^{-1}DZ'X
	XtiVX = XtX - XtZ @ np.linalg.inv(I + D @ ZtZ) @ D @ ZtX

	# New empty array for differentiating S^2 wrt gamma.
	dS2dgamma = np.array([])

	# Work of derivative wrt to sigma^2 
	dS2dsigma2 = 

	# Add to 

	for k in np.arange(len(nparams)):

		dS2dVechDk = np.zeros#...

    	for j in np.arange(nlevels[k]):

    		# Get the indices for this level and factor.
    		Ikj = faclev_indices2D(k, j, nlevels, nparams)
				    
		    # Work out Z_(k,j)'e
		    ZkjtZ = ZtZ[:, Ikj,:]

		    # Work out Z_(k,j)'V^{-1}X
		    Z

			dS2dvechDk = dS2dvechDk + mat2vech2D#(BLAH)



