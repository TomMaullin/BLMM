import numpy as np
import os
import time
import scipy.sparse
import scipy.sparse.linalg
import sys
import nibabel as nib
import nilearn
from lib.tools3d import *
from lib.tools2d import *
import sparse
#from lib.FS import FS
#from lib.SFS import SFS
#from lib.pFS import pFS
#from lib.pSFS import pSFS
from lib.FS2D import FS2D
from lib.pFS2D import pFS2D
from lib.PLS import PLS2D, PLS2D_getBeta, PLS2D_getD, PLS2D_getSigma2
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack
from scipy.optimize import minimize

# Random Field based simulation
def main():

    #================================================================================
    # Scalars
    #================================================================================
    # Number of factors, random integer between 1 and 3
    r = 2#np.random.randint(2,4)#np.random.randint(1,4)
    #print("Number of grouping factors for random effects:")
    #print(r)

    # Number of levels, random number between 2 and 8
    nlevels = np.array([30,10])#np.random.randint(2,8,size=(r))
    # Let the first number of levels be a little larger (typically like subjects)
    #nlevels[0] = np.random.randint(2,35,size=1)
    #nlevels = np.sort(nlevels)[::-1]
    #print("Number of levels for each factor:")
    #print(nlevels)

    # Number of parameters, random number between 1 and 5
    nparams = np.array([3,1])#np.random.randint(1,6,size=(r))
    #print("Number of parameters for each factor:")
    #print(nparams)

    # Dimension of D
    #print("Dimension of D, q:")
    q = np.sum(nlevels*nparams)
    #print(q)

    # Number of fixed effects, random number between 6 and 30
    p = 5#np.random.randint(6,31)
    #print("Number of fixed effects:")
    #print(p)

    # Number of subjects, n
    n = 1000
    #print("Number of subjects:")
    #print(n)

    # Voxel dimensions
    dimv = [20,20,20]
    nv = np.prod(dimv)
    #print("Number of voxels:")
    #print(nv)

    #================================================================================
    # Design matrix
    #================================================================================
    # Initialize empty x
    X = np.zeros((n,p))

    # First column is intercept
    X[:,0] = 1

    # Rest of the columns we will make random noise 
    X[:,1:] = np.random.randn(n*(p-1)).reshape((n,(p-1)))

    #================================================================================
    # Random Effects Design matrix
    #================================================================================
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

    #================================================================================
    # Smoothed beta
    #================================================================================
    # Random 4D matrix (unsmoothed)
    beta_us = np.random.randn(nv*p).reshape(dimv[0],dimv[1],dimv[2],p)*20
    beta_us[3:5,3:5,3:5,3] = beta_us[3:5,3:5,3:5,3] + 100

    t1 = time.time()
    # Some random affine, not important for this simulation
    affine = np.diag([1, 1, 1, 1])
    beta_us_nii = nib.Nifti1Image(beta_us, affine)

    # Smoothed beta nifti
    beta_s_nii = nilearn.image.smooth_img(beta_us_nii, 5)

    # Final beta
    beta = beta_s_nii.get_fdata()


    #================================================================================
    # Smoothed b
    #================================================================================
    # Random 4D matrix (unsmoothed)
    b_us = np.random.randn(nv*q).reshape(dimv[0],dimv[1],dimv[2],q)*20

    # Some random affine, not important for this simulation
    affine = np.diag([1, 1, 1, 1])
    b_us_nii = nib.Nifti1Image(b_us, affine)

    # Smoothed beta nifti
    b_s_nii = nilearn.image.smooth_img(b_us_nii, 5)

    # Final beta
    b = b_s_nii.get_fdata()

    #================================================================================
    # Response
    #================================================================================
    # Reshape X
    X = X.reshape(1, X.shape[0], X.shape[1])

    # Reshape beta
    beta = beta.reshape(beta.shape[0]*beta.shape[1]*beta.shape[2],beta.shape[3],1)
    beta_True = beta

    # Reshape Z (note: This step is slow because of the sparse to dense conversion;
    # it could probably be made quicker but this is only for one simulation at current)
    Ztmp = Z.toarray().reshape(1, Z.shape[0], Z.shape[1])

    # Reshape b
    b = b.reshape(b.shape[0]*b.shape[1]*b.shape[2],b.shape[3],1)

    # Generate Y
    Y = np.matmul(X,beta)+np.matmul(Ztmp,b) + np.random.randn(n,1)



    #================================================================================
    # Transpose products
    #================================================================================
    # X'Z\Z'X
    XtZ = np.matmul(X.transpose(0,2,1),Ztmp)
    ZtX = XtZ.transpose(0,2,1)

    # Z'Y\Y'Z
    YtZ = np.matmul(Y.transpose(0,2,1),Ztmp)
    ZtY = YtZ.transpose(0,2,1)

    # Y'X/X'Y
    YtX = np.matmul(Y.transpose(0,2,1),X)
    XtY = YtX.transpose(0,2,1)

    # YtY
    YtY = np.matmul(Y.transpose(0,2,1),Y)

    # ZtZ
    ZtZ = np.matmul(Ztmp.transpose(0,2,1),Ztmp)

    # X'X
    XtX = np.matmul(X.transpose(0,2,1),X)

    mode = 'pFS'

    if mode == 'PLS':

        #================================================================================
        # Initial theta
        #================================================================================
        theta0 = np.array([])
        for i in np.arange(r):
          theta0 = np.hstack((theta0, mat2vech2D(np.eye(nparams[i])).reshape(np.int64(nparams[i]*(nparams[i]+1)/2))))
      
        #================================================================================
        # Indices required by DAC
        #================================================================================
        inds = np.arange(nv).reshape(dimv)
        tinds,rinds,cinds=get_mapping2D(nlevels, nparams)
        Lam=mapping2D(np.random.randn(theta0.shape[0]),tinds,rinds,cinds)

        # Obtain Lambda'Z'ZLambda
        LamtZtZLam = spmatrix.trans(Lam)*cvxopt.sparse(matrix(ZtZ[0,:,:]))*Lam

        # Obtaining permutation for PLS
        cholmod.options['supernodal']=2
        P=amd.order(LamtZtZLam)

        # Identity (Actually quicker to calculate outside of estimation)
        I = spmatrix(1.0, range(Lam.size[0]), range(Lam.size[0]))

        # New array we will store estimates in
        est_theta = np.zeros((XtY.shape[0], theta0.shape[0]))

        print('original est_theta shape')
        print(est_theta.shape)

        #================================================================================
        # Run Simulation
        #================================================================================
        t1 = time.time()
        est_theta = divAndConq_PLS(theta0, inds, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        t2 = time.time()
        print('Time taken (seconds):', t2-t1)

        #================================================================================
        # Performance metric for beta
        #================================================================================
        # See how well it did
        #beta_True_map=beta_True.reshape(dimv[0],dimv[1],dimv[2],beta.shape[1])
        #beta_est_map=paramVec[:,0:p,:].reshape(dimv[0],dimv[1],dimv[2],beta.shape[1])
        #print(np.mean(np.mean(np.mean(np.abs(beta_True_map-beta_est_map)))))

        beta_est = np.zeros(beta_True.shape)
        # Get betas from theta estimate
        XtX_current = cvxopt.matrix(XtX[0,:,:])
        XtZ_current = cvxopt.matrix(XtZ[0,:,:])
        ZtX_current = cvxopt.matrix(ZtX[0,:,:])
        ZtZ_current = cvxopt.sparse(cvxopt.matrix(ZtZ[0,:,:]))
        beta_runningsum = 0
        b_runningsum = 0
        for i in np.arange(est_theta.shape[0]):
            theta = est_theta[i,:]
            XtY_current = cvxopt.matrix(XtY[i,:,:])
            YtX_current = cvxopt.matrix(YtX[i,:,:])
            YtY_current = cvxopt.matrix(YtY[i,:,:])
            YtZ_current = cvxopt.matrix(YtZ[i,:,:])
            ZtY_current = cvxopt.matrix(ZtY[i,:,:])

            beta_est = np.array(PLS2D_getBeta(theta, ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, YtZ_current, XtZ_current, YtY_current, n, P, tinds, rinds, cinds))

            sigma2_est = PLS2D_getSigma2(theta, ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, YtZ_current, XtZ_current, YtY_current, n, P, I, tinds, rinds, cinds)
            D_est = np.array(matrix(PLS2D_getD(theta, tinds, rinds, cinds, sigma2_est)))

            DinvIplusZtZD = D_est @ np.linalg.inv(np.eye(q) + np.array(ZtZ[0,:,:]) @ D_est)
            Zte = np.array(ZtY_current) - np.array(ZtX[0,:,:]) @ beta_est
            b_est = (DinvIplusZtZD @ Zte)
            b_true = b[i,:]

            beta_runningsum = beta_runningsum + np.sum(np.abs(beta_True[i,:] - beta_est))
            b_runningsum = b_runningsum + np.sum(np.abs(b_true - b_est))

        print(beta_runningsum/(nv*p))
        print(b_runningsum/(nv*q))

    if mode == 'FS':
        #================================================================================
        # Indices required by DAC
        #================================================================================
        inds = np.arange(nv).reshape(dimv)
        
        # Matrices for estimating mean of current block
        XtX_current = XtX[0,:,:]
        XtY_current = np.mean(XtY, axis=0)
        XtZ_current = XtZ[0,:,:]
        YtX_current = np.mean(YtX,axis=0)
        YtY_current = np.mean(YtY,axis=0)
        YtZ_current = np.mean(YtZ,axis=0)
        ZtX_current = ZtX[0,:,:]
        ZtY_current = np.mean(ZtY,axis=0)
        ZtZ_current = ZtZ[0,:,:]

        # Inital beta
        beta = initBeta2D(XtX_current, XtY_current)

        # Work out e'e
        ete = ssr2D(YtX_current, YtY_current, XtX_current, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)

        Zte = ZtY_current - (ZtX_current @ beta)

        # Inital D
        # Dictionary version
        Ddict = dict()
        for k in np.arange(len(nparams)):

          Ddict[k] = makeDnnd2D(initDk2D(k, nlevels[k], ZtZ_current, Zte, sigma2, nparams, nlevels))
          
        paramVector = np.concatenate((beta, np.array([[sigma2]])))

        for k in np.arange(len(nparams)):

          paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

        # New array we will store estimates in
        est_params = np.zeros((XtY.shape[0], paramVector.shape[0]))

        t1 = time.time()
        est_params = divAndConq_FS(paramVector, inds, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        t2 = time.time()
        print(t2-t1)

        #print(paramVector)
        beta = est_params[:,0:p].reshape(beta_True.shape)
        sigma2 = est_params[:,p:(p+1)][0,0]


        FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + p + 1)
        FishIndsDk = np.insert(FishIndsDk,0,p+1)

        # D as a dictionary
        for k in np.arange(len(nparams)):

            Ddict[k] = makeDnnd3D(vech2mat3D(est_params[:,FishIndsDk[k]:FishIndsDk[k+1]]))

        # Full version of D
        D = getDfromDict3D(Ddict, nparams, nlevels)

        betaDiff = np.mean(np.mean(np.mean(np.abs(beta_True - beta))))
        print(betaDiff)

        DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
        Zte = ZtY - ZtX @ beta
        b_est = (DinvIplusZtZD @ Zte).reshape(b.shape)
        bDiff = np.mean(np.mean(np.mean(np.abs(b_est - b))))

        print(bDiff)

    if mode == 'pFS':
        #================================================================================
        # Indices required by DAC
        #================================================================================
        inds = np.arange(nv).reshape(dimv)
        
        # Matrices for estimating mean of current block
        XtX_current = XtX[0,:,:]
        XtY_current = np.mean(XtY, axis=0)
        XtZ_current = XtZ[0,:,:]
        YtX_current = np.mean(YtX,axis=0)
        YtY_current = np.mean(YtY,axis=0)
        YtZ_current = np.mean(YtZ,axis=0)
        ZtX_current = ZtX[0,:,:]
        ZtY_current = np.mean(ZtY,axis=0)
        ZtZ_current = ZtZ[0,:,:]

        # Inital beta
        beta = initBeta2D(XtX_current, XtY_current)

        # Work out e'e
        ete = ssr2D(YtX_current, YtY_current, XtX_current, beta)

        # Initial sigma2
        sigma2 = initSigma22D(ete, n)

        Zte = ZtY_current - (ZtX_current @ beta)

        # Inital D
        # Dictionary version
        Ddict = dict()
        for k in np.arange(len(nparams)):

          Ddict[k] = makeDnnd2D(initDk2D(k, nlevels[k], ZtZ_current, Zte, sigma2, nparams, nlevels))
          
        paramVector = np.concatenate((beta, np.array([[sigma2]])))

        for k in np.arange(len(nparams)):

          paramVector = np.concatenate((paramVector, mat2vech2D(Ddict[k])))

        # New array we will store estimates in
        est_params = np.zeros((XtY.shape[0], paramVector.shape[0]))

        t1 = time.time()
        est_params = divAndConq_pFS(paramVector, inds, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        t2 = time.time()
        print(t2-t1)

        #print(paramVector)
        beta = est_params[:,0:p].reshape(beta_True.shape)
        sigma2 = est_params[:,p:(p+1)][0,0]


        FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + p + 1)
        FishIndsDk = np.insert(FishIndsDk,0,p+1)

        # D as a dictionary
        for k in np.arange(len(nparams)):

            Ddict[k] = makeDnnd3D(vech2mat3D(est_params[:,FishIndsDk[k]:FishIndsDk[k+1]]))

        # Full version of D
        D = getDfromDict3D(Ddict, nparams, nlevels)

        betaDiff = np.mean(np.mean(np.mean(np.abs(beta_True - beta))))
        print(betaDiff)

        DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
        Zte = ZtY - ZtX @ beta
        b_est = (DinvIplusZtZD @ Zte).reshape(b.shape)
        bDiff = np.mean(np.mean(np.mean(np.abs(b_est - b))))

        print(bDiff)


def divAndConq_PLS(init_theta, current_inds, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta):

    # Number of voxels and dimension of block we are looking at
    current_dimv = current_inds.shape
    current_nv = np.prod(current_dimv)

    # Current indices as a vector
    current_inds_vec = current_inds.reshape(current_nv)

    # Matrices for estimating mean of current block
    XtX_current = cvxopt.matrix(XtX[0,:,:])
    XtY_current = cvxopt.matrix(np.mean(XtY[current_inds_vec,:,:], axis=0))
    XtZ_current = cvxopt.matrix(XtZ[0,:,:])
    YtX_current = cvxopt.matrix(np.mean(YtX[current_inds_vec,:,:],axis=0))
    YtY_current = cvxopt.matrix(np.mean(YtY[current_inds_vec,:,:],axis=0))
    YtZ_current = cvxopt.matrix(np.mean(YtZ[current_inds_vec,:,:],axis=0))
    ZtX_current = cvxopt.matrix(ZtX[0,:,:])
    ZtY_current = cvxopt.matrix(np.mean(ZtY[current_inds_vec,:,:],axis=0))
    ZtZ_current = cvxopt.sparse(cvxopt.matrix(ZtZ[0,:,:]))

    # Get new theta
    tmp = minimize(PLS2D, init_theta, args=(ZtX_current, ZtY_current, XtX_current, ZtZ_current, XtY_current, YtX_current, YtZ_current, XtZ_current, YtY_current, n, P, I, tinds, rinds, cinds), method='L-BFGS-B', tol=1e-7)
    new_theta = tmp.x

    if current_dimv[0]!=1 and current_dimv[1]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block4 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block5 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block6 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block7 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,(current_dimv[2]//2):]
        current_inds_block8 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,(current_dimv[2]//2):]

        est_theta = divAndConq_PLS(new_theta, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block5, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block6, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block7, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block8, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)

    elif current_dimv[0]!=1 and current_dimv[1]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),:]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),:]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,:]
        current_inds_block4 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,:]

        est_theta = divAndConq_PLS(new_theta, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)

    elif current_dimv[0]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:,:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:,:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),:,(current_dimv[2]//2):]
        current_inds_block4 = current_inds[(current_dimv[0]//2):,:,(current_dimv[2]//2):]

        est_theta = divAndConq_PLS(new_theta, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)


    elif current_dimv[1]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:,:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[:,(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:,:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block4 = current_inds[:,(current_dimv[1]//2):,(current_dimv[2]//2):]

        est_theta = divAndConq_PLS(new_theta, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)

    elif current_dimv[0]!=1:

        current_inds_block1 = current_inds[:(current_dimv[0]//2),:,:]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:,:]

        est_theta = divAndConq_PLS(new_theta, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)

    elif current_dimv[1]!=1:

        current_inds_block1 = current_inds[:,:(current_dimv[1]//2),:]
        current_inds_block2 = current_inds[:,(current_dimv[1]//2):,:]

        est_theta = divAndConq_PLS(new_theta, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)

    elif current_dimv[2]!=1:

        current_inds_block1 = current_inds[:,:,:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[:,:,(current_dimv[2]//2):]

        est_theta = divAndConq_PLS(new_theta, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)
        est_theta = divAndConq_PLS(new_theta, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds, est_theta)

    else:

        # Save parameter estimates in correct location if we are only looking at one voxel
        est_theta[current_inds[:],:] = new_theta

    return(est_theta)


def divAndConq_FS(init_params, current_inds, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params):

    # Number of voxels and dimension of block we are looking at
    current_dimv = current_inds.shape
    current_nv = np.prod(current_dimv)

    # Current indices as a vector
    current_inds_vec = current_inds.reshape(current_nv)

    # Matrices for estimating mean of current block
    XtX_current = XtX[0,:,:]
    XtY_current = np.mean(XtY[current_inds_vec,:,:], axis=0)
    XtZ_current = XtZ[0,:,:]
    YtX_current = np.mean(YtX[current_inds_vec,:,:],axis=0)
    YtY_current = np.mean(YtY[current_inds_vec,:,:],axis=0)
    YtZ_current = np.mean(YtZ[current_inds_vec,:,:],axis=0)
    ZtX_current = ZtX[0,:,:]
    ZtY_current = np.mean(ZtY[current_inds_vec,:,:],axis=0)
    ZtZ_current = ZtZ[0,:,:]

    # Get new params
    tmp = FS2D(XtX_current, XtY_current, ZtX_current, ZtY_current, ZtZ_current, XtZ_current, YtZ_current, YtY_current, YtX_current, nlevels, nparams, 1e-6, n, init_params)
    new_params = tmp[0]

    if current_dimv[0]!=1 and current_dimv[1]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block4 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block5 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block6 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block7 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,(current_dimv[2]//2):]
        current_inds_block8 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block5, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block6, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block7, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block8, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[0]!=1 and current_dimv[1]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),:]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),:]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,:]
        current_inds_block4 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,:]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[0]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:,:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:,:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),:,(current_dimv[2]//2):]
        current_inds_block4 = current_inds[(current_dimv[0]//2):,:,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)


    elif current_dimv[1]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:,:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[:,(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:,:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block4 = current_inds[:,(current_dimv[1]//2):,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[0]!=1:

        current_inds_block1 = current_inds[:(current_dimv[0]//2),:,:]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:,:]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[1]!=1:

        current_inds_block1 = current_inds[:,:(current_dimv[1]//2),:]
        current_inds_block2 = current_inds[:,(current_dimv[1]//2):,:]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[2]!=1:

        current_inds_block1 = current_inds[:,:,:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[:,:,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    else:

        # Save parameter estimates in correct location if we are only looking at one voxel
        est_params[current_inds[:],:] = new_params.reshape(est_params[current_inds[:],:].shape)

    return(est_params)



def divAndConq_pFS(init_params, current_inds, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params):

    # Number of voxels and dimension of block we are looking at
    current_dimv = current_inds.shape
    current_nv = np.prod(current_dimv)

    # Current indices as a vector
    current_inds_vec = current_inds.reshape(current_nv)

    # Matrices for estimating mean of current block
    XtX_current = XtX[0,:,:]
    XtY_current = np.mean(XtY[current_inds_vec,:,:], axis=0)
    XtZ_current = XtZ[0,:,:]
    YtX_current = np.mean(YtX[current_inds_vec,:,:],axis=0)
    YtY_current = np.mean(YtY[current_inds_vec,:,:],axis=0)
    YtZ_current = np.mean(YtZ[current_inds_vec,:,:],axis=0)
    ZtX_current = ZtX[0,:,:]
    ZtY_current = np.mean(ZtY[current_inds_vec,:,:],axis=0)
    ZtZ_current = ZtZ[0,:,:]

    # Get new params
    tmp = pFS2D(XtX_current, XtY_current, ZtX_current, ZtY_current, ZtZ_current, XtZ_current, YtZ_current, YtY_current, YtX_current, nlevels, nparams, 1e-6, n, init_params)
    new_params = tmp[0]

    if current_dimv[0]!=1 and current_dimv[1]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block4 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block5 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block6 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block7 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,(current_dimv[2]//2):]
        current_inds_block8 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block5, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block6, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block7, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block8, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[0]!=1 and current_dimv[1]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:(current_dimv[1]//2),:]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:(current_dimv[1]//2),:]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),(current_dimv[1]//2):,:]
        current_inds_block4 = current_inds[(current_dimv[0]//2):,(current_dimv[1]//2):,:]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[0]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:(current_dimv[0]//2),:,:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:,:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:(current_dimv[0]//2),:,(current_dimv[2]//2):]
        current_inds_block4 = current_inds[(current_dimv[0]//2):,:,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)


    elif current_dimv[1]!=1 and current_dimv[2]!=1:

        # Split into blocks - assuming current inds is a block
        current_inds_block1 = current_inds[:,:(current_dimv[1]//2),:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[:,(current_dimv[1]//2):,:(current_dimv[2]//2)]
        current_inds_block3 = current_inds[:,:(current_dimv[1]//2),(current_dimv[2]//2):]
        current_inds_block4 = current_inds[:,(current_dimv[1]//2):,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block3, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block4, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[0]!=1:

        current_inds_block1 = current_inds[:(current_dimv[0]//2),:,:]
        current_inds_block2 = current_inds[(current_dimv[0]//2):,:,:]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[1]!=1:

        current_inds_block1 = current_inds[:,:(current_dimv[1]//2),:]
        current_inds_block2 = current_inds[:,(current_dimv[1]//2):,:]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    elif current_dimv[2]!=1:

        current_inds_block1 = current_inds[:,:,:(current_dimv[2]//2)]
        current_inds_block2 = current_inds[:,:,(current_dimv[2]//2):]

        est_params = divAndConq_FS(new_params, current_inds_block1, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)
        est_params = divAndConq_FS(new_params, current_inds_block2, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, nlevels, nparams, n, est_params)

    else:

        # Save parameter estimates in correct location if we are only looking at one voxel
        est_params[current_inds[:],:] = new_params.reshape(est_params[current_inds[:],:].shape)

    return(est_params)


