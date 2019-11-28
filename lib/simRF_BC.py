import numpy as np
import os
import time
import scipy.sparse
import scipy.sparse.linalg
import sys
import nibabel as nib
import nilearn
from lib.tools3d import *
import sparse
from lib.FS import FS
from lib.SFS import SFS
from lib.pFS import pFS
from lib.pSFS import pSFS

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
	paramVec = FS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
	#paramVec = pFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
	#paramVec = SFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
	#paramVec = pSFS(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nparams, 1e-6,n)
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

	  Ddict[k] = makeDnnd3D(vech2mat3D(paramVec[:,FishIndsDk[k]:FishIndsDk[k+1],:]))

	  #Ddict[k] = makeDnnd3D(vech2mat3D(paramVec[:,FishIndsDk[k]:FishIndsDk[k+1],:]))


	  
	# Full version of D
	D = getDfromDict3D(Ddict, nparams, nlevels)

	DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

	Zte = ZtY - ZtX @ beta

	    
	b_est = (DinvIplusZtZD @ Zte).reshape(dimv[0],dimv[1],dimv[2],q)
	b_true = b.reshape(dimv[0],dimv[1],dimv[2],q)
	print(np.mean(np.mean(np.mean(np.abs(b_true-b_est)))))