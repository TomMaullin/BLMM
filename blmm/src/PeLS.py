import numpy as np
import cvxopt
from cvxopt import cholmod, matrix, spmatrix, lapack
from blmm.src.npMatrix2d import *
from blmm.src.cvxMatrix2d import *

def PeLS2D(theta, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds):

    # Obtain Lambda
    Lambda = mapping2D(theta, tinds, rinds, cinds)
    
    # Obtain Lambda'
    Lambdat = spmatrix.trans(Lambda)

    # Obtain Lambda'Z'Y and Lambda'Z'X
    LambdatZtY = Lambdat*ZtY
    LambdatZtX = Lambdat*ZtX
    
    # Obtain the cholesky decomposition
    LambdatZtZLambda = Lambdat*(ZtZ*Lambda)
    chol_dict = sparse_chol2D(LambdatZtZLambda+I, perm=P, retF=True, retP=False, retL=False)
    F = chol_dict['F']

    # Obtain C_u (annoyingly solve writes over the second argument,
    # whereas spsolve outputs)
    Cu = LambdatZtY[P,:]
    cholmod.solve(F,Cu,sys=4)

    # Obtain RZX
    RZX = LambdatZtX[P,:]
    cholmod.solve(F,RZX,sys=4)

    # Obtain RXtRX
    RXtRX = XtX - matrix.trans(RZX)*RZX

    # Obtain beta estimates (note: gesv also replaces the second
    # argument)
    betahat = XtY - matrix.trans(RZX)*Cu
    try:
        lapack.posv(RXtRX, betahat)
    except:
        lapack.gesv(RXtRX, betahat)

    # Obtain u estimates
    uhat = Cu-RZX*betahat
    cholmod.solve(F,uhat,sys=5)
    cholmod.solve(F,uhat,sys=8)

    # Obtain b estimates
    bhat = Lambda*uhat
    
    # Obtain residuals sum of squares
    resss = YtY-2*YtX*betahat-2*YtZ*bhat+2*matrix.trans(betahat)*XtZ*bhat+matrix.trans(betahat)*XtX*betahat+matrix.trans(bhat)*ZtZ*bhat
    
    # Obtain penalised residual sum of squares
    pss = resss + matrix.trans(uhat)*uhat
    
    # Obtain Log(|L|^2)
    logdet = 2*sum(cvxopt.log(cholmod.diag(F)))
    
    # Obtain log likelihood
    logllh = -logdet/2-n/2*(1+np.log(2*np.pi*pss[0,0])-np.log(n))
    
    return(-logllh)

def PeLS2D_getBeta(theta, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, tinds, rinds, cinds):

    # Obtain Lambda
    Lambda = mapping2D(theta, tinds, rinds, cinds)
    
    # Obtain Lambda'
    Lambdat = spmatrix.trans(Lambda)

    # Obtain Lambda'Z'Y and Lambda'Z'X
    LambdatZtY = Lambdat*ZtY
    LambdatZtX = Lambdat*ZtX
    
    # Set the factorisation to use LL' instead of LDL'
    cholmod.options['supernodal']=2

    # Obtain the cholesky decomposition
    LambdatZtZLambda = Lambdat*ZtZ*Lambda
    I = spmatrix(1.0, range(Lambda.size[0]), range(Lambda.size[0]))
    chol_dict = sparse_chol2D(LambdatZtZLambda+I, perm=P, retF=True, retP=False, retL=False)
    F = chol_dict['F']

    # Obtain C_u (annoyingly solve writes over the second argument,
    # whereas spsolve outputs)
    Cu = LambdatZtY[P,:]
    cholmod.solve(F,Cu,sys=4)

    # Obtain RZX
    RZX = LambdatZtX[P,:]
    cholmod.solve(F,RZX,sys=4)

    # Obtain RXtRX
    RXtRX = XtX - matrix.trans(RZX)*RZX

    # Obtain beta estimates (note: gesv also replaces the second
    # argument)
    betahat = XtY - matrix.trans(RZX)*Cu
    try:
        lapack.posv(RXtRX, betahat)
    except:
        lapack.gesv(RXtRX, betahat)

    return(betahat)



def PeLS2D_getSigma2(theta, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, n, P, I, tinds, rinds, cinds):

    # Obtain Lambda
    #t1 = time.time()
    Lambda = mapping2D(theta, tinds, rinds, cinds)
    #t2 = time.time()
    #print(t2-t1)#3.170967102050781e-05   9
    
    # Obtain Lambda'
    #t1 = time.time()
    Lambdat = spmatrix.trans(Lambda)
    #t2 = time.time()
    #print(t2-t1)# 3.5762786865234375e-06

    # Obtain Lambda'Z'Y and Lambda'Z'X
    #t1 = time.time()
    LambdatZtY = Lambdat*ZtY
    LambdatZtX = Lambdat*ZtX
    #t2 = time.time()
    #print(t2-t1)#1.049041748046875e-05   13
    
    # Obtain the cholesky decomposition
    #t1 = time.time()
    LambdatZtZLambda = Lambdat*(ZtZ*Lambda)
    #t2 = time.time()
    #print(t2-t1)#3.790855407714844e-05   2
    
    #t1 = time.time()
    chol_dict = sparse_chol2D(LambdatZtZLambda+I, perm=P, retF=True, retP=False, retL=False)
    F = chol_dict['F']
    #t2 = time.time()
    #print(t2-t1)#0.0001342296600341797   1

    # Obtain C_u (annoyingly solve writes over the second argument,
    # whereas spsolve outputs)
    #t1 = time.time()
    Cu = LambdatZtY[P,:]
    cholmod.solve(F,Cu,sys=4)
    #t2 = time.time()
    #print(t2-t1)#1.5974044799804688e-05   5

    # Obtain RZX
    #t1 = time.time()
    RZX = LambdatZtX[P,:]
    cholmod.solve(F,RZX,sys=4)
    #t2 = time.time()
    #print(t2-t1)#1.2159347534179688e-05   7

    # Obtain RXtRX
    #t1 = time.time()
    RXtRX = XtX - matrix.trans(RZX)*RZX
    #t2 = time.time()
    #print(t2-t1)#9.775161743164062e-06  11

    # Obtain beta estimates (note: gesv also replaces the second
    # argument)
    #t1 = time.time()
    betahat = XtY - matrix.trans(RZX)*Cu
    try:
        lapack.posv(RXtRX, betahat)
    except:
        lapack.gesv(RXtRX, betahat)
    #t2 = time.time()
    #print(t2-t1)#1.7404556274414062e-05   6

    # Obtain u estimates
    #t1 = time.time()
    uhat = Cu-RZX*betahat
    cholmod.solve(F,uhat,sys=5)
    cholmod.solve(F,uhat,sys=8)
    #t2 = time.time()
    #print(t2-t1)#1.2874603271484375e-05   8
    
    # Obtain b estimates
    #t1 = time.time()
    bhat = Lambda*uhat
    #t2 = time.time()
    #print(t2-t1)#2.86102294921875e-06  15
    
    # Obtain residuals sum of squares
    #t1 = time.time()
    resss = YtY-2*YtX*betahat-2*YtZ*bhat+2*matrix.trans(betahat)*XtZ*bhat+matrix.trans(betahat)*XtX*betahat+matrix.trans(bhat)*ZtZ*bhat
    #t2 = time.time()
    #print(t2-t1)#3.409385681152344e-05   4
    
    # Obtain penalised residual sum of squares
    #t1 = time.time()
    pss = resss + matrix.trans(uhat)*uhat

    return(pss/n)

def PeLS2D_getD(theta, tinds, rinds, cinds, sigma2):

    # Obtain Lambda
    Lambda = mapping2D(theta, tinds, rinds, cinds)
    
    # Obtain Lambda'
    Lambdat = spmatrix.trans(Lambda)

    # Get D
    D = Lambda * Lambdat * sigma2

    return(D)