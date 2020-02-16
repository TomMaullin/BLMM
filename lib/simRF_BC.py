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
from lib.FS import FS
from lib.SFS import SFS
from lib.pFS import pFS
from lib.pSFS import pSFS
from lib.sattherthwaite import *
import cvxopt

# Random Field based simulation
def main():

	t1 = time.time()

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
	dimv = [5,5,5]
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
			#Zdata_factor[:,0]=1

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
	beta_us = np.random.randn(nv*p).reshape(dimv[0],dimv[1],dimv[2],p)
	#beta_us[3:5,3:5,3:5,3] = beta_us[3:5,3:5,3:5,3] + 100

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
	b_us = np.random.randn(nv*q).reshape(dimv[0],dimv[1],dimv[2],q)

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

	#================================================================================
	# Run Simulation
	#================================================================================
	t1 = time.time()
	#paramVec = FS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
	#paramVec = pFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
	#paramVec = SFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
	paramVec = pSFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
	t2 = time.time()
	print(t2-t1)

	#================================================================================
	# Performance metric for beta
	#================================================================================
	# See how well it did
	beta_True_map=beta_True.reshape(dimv[0],dimv[1],dimv[2],beta.shape[1])
	beta_est_map=paramVec[:,0:p,:].reshape(dimv[0],dimv[1],dimv[2],beta.shape[1])
	print(np.mean(np.mean(np.mean(np.abs(beta_True_map-beta_est_map)))))


	#FishIndsDk = np.int32(np.cumsum(nparams**2) + p + 1)
	FishIndsDk = np.int32(np.cumsum(nparams*(nparams+1)/2) + p + 1)
	FishIndsDk = np.insert(FishIndsDk,0,p+1)

	# Get the parameters
	beta = paramVec[:,0:p,:]
	sigma2 = paramVec[:,p:(p+1)][:,0,0]

	Ddict = dict()
	# D as a dictionary
	for k in np.arange(len(nparams)):

	  #Ddict[k] = makeDnnd3D(vech2mat3D(paramVec[:,FishIndsDk[k]:FishIndsDk[k+1],:]))

	  Ddict[k] = makeDnnd3D(vech2mat3D(paramVec[:,FishIndsDk[k]:FishIndsDk[k+1],:]))

	  
	# Full version of D
	D = getDfromDict3D(Ddict, nparams, nlevels)

	DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

	Zte = ZtY - ZtX @ beta

	    
	b_est = (DinvIplusZtZD @ Zte).reshape(dimv[0],dimv[1],dimv[2],q)
	b_true = b.reshape(dimv[0],dimv[1],dimv[2],q)
	print(np.mean(np.mean(np.mean(np.abs(b_true-b_est)))))


	#================================================================================
	# Theta representation of D, sigma^2 and beta - NTS MAKE SEPERATE FN
	#================================================================================

	# Initiate empty theta
	theta = np.zeros((D.shape[0], np.int32(np.sum(nparams*(nparams+1)/2))))

	# Indices for D/lambda submatrices
	Dinds = np.insert(np.cumsum(nlevels*nparams),0,0)

	# Get 3D theta representation
	for i in np.arange(D.shape[0]):

		# # Look at individual D
		# Di = cvxopt.sparse(matrix(D[i,:,:]))

		# # Perform sparse cholesky on D.
		# try:
		# 	chol_dict = sparse_chol2D(Di, perm=None, retF=True, retP=False, retL=True)
		# except:
		# 	print('catch active')
		# 	print(type(Di))
		# 	print(Di.size)
		# 	print(Di)

		# Lami = chol_dict['L']
		# Lami = np.array(matrix(Lami))

		# Unique elements of lambda
		vecuLami = np.array([])

		for j in np.arange(len(nparams)):

			Dij = D[i,Dinds[j]:(Dinds[j]+nparams[j]),Dinds[j]:(Dinds[j]+nparams[j])]

			try:
				Lamij = np.linalg.cholesky(Dij)
			except:
				L, Dvals, perm = scipy.linalg.ldl(Dij)
				Lamij = np.real(np.matmul(L[perm,:], np.sqrt(Dvals+0J)))
				# print('L')
				# print(L[perm,:])
				# print('Dvals')
				# print(Dvals)
				# print('Lam')
				# print(Lamij)
				# print('difference')
				# print(Lamij @ Lamij.transpose() - Dij)

			# Get individual block of lambda
			#Lamij = Lami[Dinds[j]:(Dinds[j]+nparams[j]),Dinds[j]:(Dinds[j]+nparams[j])]

			# Convert it to vec(h) format
			vecuLamij = mat2vech2D(Lamij)

			# Add it to running element list
			vecuLami = np.concatenate((vecuLami, vecuLamij), axis=None)

		# Look at individual sigma2
		sigma2i = sigma2[i]

		# Compose theta vector
		thetai = vecuLami*np.sqrt(sigma2i)

		# Add theta vector to array
		theta[i,:] = thetai.reshape(theta[i,:].shape)

	print(theta.shape)

	#================================================================================
	# Sattherthwaite degrees of freedom (BLMM version)
	#================================================================================

	print('up to SattherthwaiteDoF BLMM method')
	print('time elapsed prior: ', time.time()-t1)
	t2 = time.time()

	# Get L contrast
	L = np.zeros((1,p))
	L[0,3] = 1

	df = SattherthwaiteDoF('T','BLMM',D,sigma2,L,ZtX,ZtY,XtX,ZtZ,XtY,YtX,YtZ,XtZ,YtY,n,nlevels,nparams,theta)

	print('df results')
	print(df.shape)
	print(np.mean(df))

	print(np.max(df))

	print(np.min(df))

	print('df[0]')
	print(df[0])

	print('SattherthwaiteDoF (BLMM) done')
	print('elapsed time: ', time.time()-t2)

	#================================================================================
	# Sattherthwaite degrees of freedom (lmertest version)
	#================================================================================

	print('up to SattherthwaiteDoF lmertest method')
	print('time elapsed prior: ', time.time()-t1)
	t2 = time.time()

	# Get L contrast
	L = np.zeros((1,p))
	L[0,3] = 1

	df = SattherthwaiteDoF('T','lmerTest',D,sigma2,L,ZtX,ZtY,XtX,ZtZ,XtY,YtX,YtZ,XtZ,YtY,n,nlevels,nparams,theta)

	print('df results')
	print(df.shape)
	print(np.mean(df))

	print(np.max(df))

	print(np.min(df))

	print('df[0]')
	print(df[0])
	
	print('SattherthwaiteDoF done')
	print('elapsed time: ', time.time()-t2)
