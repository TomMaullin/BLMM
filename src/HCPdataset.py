import pandas as pd
import numpy as np
from lib.npMatrix2d import *
from src.ADE import pFS_ADE2D
from scipy.optimize import minimize

# Read in family type dataset
#famTypewoDZ = pd.read_csv('/home/tommaullin/Documents/BLMM_creation/hcpFamTypeswithoutDZ.csv')
famTypewoDZ = pd.read_csv('/home/tommaullin/Documents/BLMM_creation/hcpFamTypeswithoutDZPMAT.csv')

# Remove columns not of interest.
reducedData = famTypewoDZ[['Subject','familyType','familyID']].sort_values(by=['familyType','familyID'])

# Read in restricted data
restricted = pd.read_csv('/home/tommaullin/Documents/BLMM_creation/RESTRICTED_nicholst_5_8_2018_15_34_43.csv')

# Reduce the restricted dataset
reducedRestricted = restricted[['Subject','Age_in_Yrs','HasGT','ZygosityGT','ZygositySR', 'Family_ID','Mother_ID','Father_ID']]

# Make a working table
newTab = pd.merge(reducedData, reducedRestricted, on='Subject')

# Get a table where every pair of parents has a corresponding count/number of children
parentTable = newTab.groupby(['Mother_ID','Father_ID']).size().sort_values(ascending=True).reset_index().rename(columns={0:'ParentCounts'})

# Add the parent counts to the new table
newTab = pd.merge(newTab,parentTable,on=['Mother_ID','Father_ID'])

# Import unrestricted data
unrestricted = pd.read_csv('/home/tommaullin/Documents/BLMM_creation/unrestricted_nicholst_4_21_2020_8_30_43.csv')

# Reduce the unrestricted Table to what we need
reducedUnrestricted = unrestricted[['Subject','Gender','ReadEng_Unadj','ReadEng_AgeAdj','FS_Total_GM_Vol','FS_IntraCranial_Vol','FS_L_Hippo_Vol','FS_R_Hippo_Vol','PMAT24_A_CR']]

# Total Hippo
reducedUnrestricted['FS_Total_Hippo_Vol'] = reducedUnrestricted['FS_L_Hippo_Vol'] + reducedUnrestricted['FS_R_Hippo_Vol']

# Demean and rescale FS_Total_GM_Vol, FS_Intracranial_Vol, FS_L_Hippo_Vol, FS_R_Hippo_Vol
#reducedUnrestricted.loc[:,'FS_Total_GM_Vol'] = (reducedUnrestricted['FS_Total_GM_Vol'] - reducedUnrestricted['FS_Total_GM_Vol'].mean())/reducedUnrestricted['FS_Total_GM_Vol'].std()
#reducedUnrestricted.loc[:,'FS_IntraCranial_Vol'] = (reducedUnrestricted['FS_IntraCranial_Vol'] - reducedUnrestricted['FS_IntraCranial_Vol'].mean())/reducedUnrestricted['FS_IntraCranial_Vol'].std()


reducedUnrestricted.loc[:,'FS_L_Hippo_Vol'] = reducedUnrestricted.loc[:,'FS_L_Hippo_Vol']**(1/3)
reducedUnrestricted.loc[:,'FS_IntraCranial_Vol'] = reducedUnrestricted.loc[:,'FS_IntraCranial_Vol']**(1/3)
#reducedUnrestricted.loc[:,'FS_L_Hippo_Vol'] = (reducedUnrestricted['FS_L_Hippo_Vol'] - reducedUnrestricted['FS_L_Hippo_Vol'].mean())/reducedUnrestricted['FS_L_Hippo_Vol'].std()
#reducedUnrestricted.loc[:,'FS_R_Hippo_Vol'] = (reducedUnrestricted['FS_R_Hippo_Vol'] - reducedUnrestricted['FS_R_Hippo_Vol'].mean())/reducedUnrestricted['FS_R_Hippo_Vol'].std()
#reducedUnrestricted.loc[:,'FS_Total_Hippo_Vol'] = (reducedUnrestricted['FS_Total_Hippo_Vol'] - reducedUnrestricted['FS_Total_Hippo_Vol'].mean())/reducedUnrestricted['FS_Total_Hippo_Vol'].std()


# Add unrestricted into table and drop na values
newTab = pd.merge(newTab,reducedUnrestricted,on=['Subject']).dropna()

#.sort_values(by=['ZygosityGT'],ascending=False)
newTab['Gender'].replace(['F','M'],[0,1],inplace=True)

# Add age and sex interaction
newTab['Age:Sex']=newTab[['Age_in_Yrs']].values*newTab[['Gender']].values

# Apply the appropriate sort
newTab=newTab.sort_values(by=['familyType','familyID','ParentCounts','ZygosityGT','ZygositySR'],ascending=[True,True,False,True,False])

# Construct X
X = newTab[['Age_in_Yrs','Gender','Age:Sex','FS_IntraCranial_Vol','ReadEng_AgeAdj']].values 

# Add an intercept to X
X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

# Construct Y # ReadEng_AgeAdj ~ 1 + Age_in_Yrs + Sex + Age:Sex + FS_IntraCranial_Vol + FS_Total_GM_Vol 
Y = newTab[['FS_L_Hippo_Vol']].values

# Number of fixed effects parameters p
p = X.shape[1]

# Output files for running same analysis in R
pd.DataFrame(X).to_csv('/home/tommaullin/Documents/BLMM_creation/X_HCP.csv')
pd.DataFrame(Y).to_csv('/home/tommaullin/Documents/BLMM_creation/Y_HCP.csv')

# # Kinship data
# Kinship = newTab[["Subject", "Mother_ID", "Father_ID", "ZygositySR", "ZygosityGT"]]

# # Recode MZ and DZ
# MZarray = ((Kinship['ZygositySR']=='MZ') | (Kinship['ZygosityGT']=='MZ')).values
# DZarray = ((Kinship['ZygositySR']=='DZ') | (Kinship['ZygosityGT']=='DZ')).values

# # Indicator column for DZ and MZ (useful for generating kinships in R)
# twinVals = np.zeros(MZarray.shape)
# twinVals[MZarray]=1
# twinVals[DZarray]=2

# # Set twinVals column
# Kinship = Kinship.assign(twinVals=twinVals)

# # Output to csv
# Kinship.to_csv('/home/tommaullin/Documents/BLMM_creation/Kin_HCP.csv',index=False)

# Construct Z
Z = np.eye(X.shape[0])

# Work out nlevels and nraneffs
UniqueFamilyTypes, idx = np.unique(newTab[['familyType']], return_index=True)
UniqueFamilyTypes = UniqueFamilyTypes[np.argsort(idx)]

# Dictionary to store Kinship matrices
KinshipA = dict()
KinshipC = dict()
KinshipD = dict()

# Number of grouping factors r
r = len(UniqueFamilyTypes)

# Number of levels and random effects for each factor
nlevels = np.zeros(r)
nraneffs = np.zeros(r)

# Loop through each family type (these are our factors)
for k in np.arange(r):
    # Record the family structure, if we haven't already.
    if k not in KinshipA:
        # Work out which family type we're looking at
        uniqueType = UniqueFamilyTypes[k]
        familyTypeTable = newTab[newTab['familyType']==uniqueType]
        # Read in the first family in this category
        uniqueFamilyIDs = np.unique(familyTypeTable[['familyID']])
        famID = uniqueFamilyIDs[0]
        famTable = familyTypeTable[familyTypeTable['familyID']==famID]
        # Work out how many subjects in family
        numSubs = len(famTable)
        # Initialize empty D_A and D_D structure
        KinshipA[k] = np.zeros((numSubs,numSubs))
        KinshipC[k] = np.ones((numSubs,numSubs))
        KinshipD[k] = np.zeros((numSubs,numSubs))
        # Loop through each pair of subjects (the families are very 
        # small in the HCP dataset so it doesn't matter if this
        # code is a little inefficient)
        for i in np.arange(numSubs):
            for j in np.arange(numSubs):
                # Check if subject i and subject j are the same person
                if i==j:
                    # In this case cov_A(i,j)=1 and cov_D(i,j)=1
                    KinshipA[k][i,j]=1
                    KinshipD[k][i,j]=1
                # Check if subject i and subject j are the MZ twins
                elif (famTable['ZygosityGT'].iloc[i]=='MZ' or famTable['ZygositySR'].iloc[i]=='MZ') and (famTable['ZygosityGT'].iloc[j]=='MZ' or famTable['ZygositySR'].iloc[j]=='MZ'):
                    # In this case cov_A(i,j)=1 and cov_D(i,j)=1
                    KinshipA[k][i,j]=1
                    KinshipD[k][i,j]=1
                # Check if subject i and subject j are full siblings (DZ is grouped into this usecase)
                elif (famTable['Mother_ID'].iloc[i]==famTable['Mother_ID'].iloc[j] and famTable['Father_ID'].iloc[i]==famTable['Father_ID'].iloc[j]):
                    # In this case cov_A(i,j)=1/2 and cov_D(i,j)=1/4
                    KinshipA[k][i,j]=1/2
                    KinshipD[k][i,j]=1/4
                # Check if subject i and subject j are half siblings
                elif (famTable['Mother_ID'].iloc[i]==famTable['Mother_ID'].iloc[j] or famTable['Father_ID'].iloc[i]==famTable['Father_ID'].iloc[j]):
                    # In this case cov_A(i,j)=1/2 and cov_D(i,j)=1/4
                    KinshipA[k][i,j]=1/4
                    KinshipD[k][i,j]=1/8
                # Else they aren't related
                else:
                    # In this case cov_A(i,j)=0 and cov_D(i,j)=0
                    KinshipA[k][i,j]=0
                    KinshipD[k][i,j]=0
        # Work out nlevels
        nlevels[k]=len(uniqueFamilyIDs)
        # Work out nraneffs
        nraneffs[k]=numSubs

# Change to ints
nlevels = np.array(nlevels, dtype=np.int32)
nraneffs = np.array(nraneffs, dtype=np.int32)

# Number of random effects, q
q = np.sum(np.dot(nraneffs,nlevels))

# Work out D indices (there is one block of D per level)
Dinds = np.zeros(np.sum(nlevels)+1)
counter = 0
for k in np.arange(len(nraneffs)):
    for j in np.arange(nlevels[k]):
        Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
        counter = counter + 1

# Last index will be missing so add it
Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]

# Make sure indices are ints
Dinds = np.int64(Dinds)


# Kinship A (matrix version)
KinshipAmat= scipy.sparse.lil_matrix((q,q))
counter = 0
for k in np.arange(len(nraneffs)):
    for j in np.arange(nlevels[k]):

        # Add a block for each level of each factor.
        KinshipAmat[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = KinshipA[k]
        counter = counter + 1

# Save Kinship A
np.savetxt("/home/tommaullin/Documents/BLMM_creation/KinshipA.csv", KinshipAmat.toarray(), delimiter=",")
del KinshipAmat


# Kinship C (matrix version)
KinshipCmat= scipy.sparse.lil_matrix((q,q))
counter = 0
for k in np.arange(len(nraneffs)):
    for j in np.arange(nlevels[k]):

        # Add a block for each level of each factor.
        KinshipCmat[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = KinshipC[k]
        counter = counter + 1

# Save Kinship A
np.savetxt("/home/tommaullin/Documents/BLMM_creation/KinshipC.csv", KinshipCmat.toarray(), delimiter=",")
del KinshipCmat


# Kinship D (matrix version)
KinshipDmat= scipy.sparse.lil_matrix((q,q))
counter = 0
for k in np.arange(len(nraneffs)):
    for j in np.arange(nlevels[k]):

        # Add a block for each level of each factor.
        KinshipDmat[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = KinshipD[k]
        counter = counter + 1

# Save Kinship D
np.savetxt("/home/tommaullin/Documents/BLMM_creation/KinshipD.csv", KinshipDmat.toarray(), delimiter=",")
del KinshipDmat


# Create structure matrices of the first kind for mapping D_k to \sigma2_A and \sigma2_B
structMat1stDict = dict()

# Loop through each family type and get the structure matrix
for k in np.arange(r):
    # Row of structure matrix k describing \sigmaA
    SkrowA = mat2vec2D(KinshipA[k]).transpose()
    # Row of structure matrix k describing \sigmaC
    SkrowC = mat2vec2D(KinshipC[k]).transpose()
    # Row of structure matrix k describing \sigmaD
    SkrowD = mat2vec2D(KinshipD[k]).transpose()
    # Construct structure matrices
    structMat1stDict[k]=np.concatenate((SkrowA,SkrowC),axis=0)

# Work out structure matrix of the second kind
structMat2nd = np.concatenate((np.tile([[1,0]],r),np.tile([[0,1]],r)),axis=0)

# Convergence tolerance
tol = 1e-3

# Number of observations
n = X.shape[0]

# Try running pFS_ADE
t1 = time.time()
tmp2=pFS_ADE2D(X, Y, nlevels, nraneffs, tol, n, KinshipA, KinshipC, structMat1stDict, structMat2nd)
t2 = time.time()
print(t2-t1)



# ================================================ #
# Scipy optimize
# ================================================ #

XtX = X.transpose() @ X
XtY = X.transpose() @ Y
YtY = Y.transpose() @ Y
YtX = Y.transpose() @ X

# ------------------------------------------------------------------------------
# Initial estimates
# ------------------------------------------------------------------------------
# If we have initial estimates use them.

# Inital beta
beta = initBeta2D(XtX, XtY)

# Work out e'e
ete = ssr2D(YtX, YtY, XtX, beta)

# Initial sigma2
sigma2 = initSigma22D(ete, n)
sigma2 = np.maximum(sigma2,1e-10) # Prevent hitting boundary
sigma2 = np.array([[sigma2]])

# Initial zero matrix to hold the matrices Skcov(dl/Dk)Sk'
FDk = np.zeros((2*r,2*r))

# Initial zero vector to hold the vectors Sk*dl/dDk
SkdldDk = np.zeros((2*r,1))

# Initial residuals
e = Y - X @ beta
eet = e @ e.transpose()

for k in np.arange(r):

    # Get FDk
    FDk[2*k:(2*k+2),2*k:(2*k+2)]= nlevels[k]*structMat1stDict[k] @ structMat1stDict[k].transpose()

    # Initialize empty sum
    eetSum = np.zeros((nraneffs[k],nraneffs[k]))

    # Get sum ee'_[k,j,j]
    for j in np.arange(nlevels[k]):

        # Get indices for current block
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

        # Add to current sum
        eetSum = eetSum + eet[np.ix_(Ikj,Ikj)]

    # Get Sk*dl/dDk
    SkdldDk[2*k:(2*k+2),:] = structMat1stDict[k] @ mat2vec2D(nlevels[k]-eetSum/sigma2)


# Initial vec(sigma^2A/sigma^2E, sigma^2D/sigma^2E)
vecAE = np.linalg.pinv(structMat2nd @ FDk @ structMat2nd.transpose()) @ structMat2nd @ SkdldDk

# Initial parameter vector
initParams = np.concatenate((beta, sigma2, vecAE*sigma2))



def llh_ADE(paramVec, X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipD):

    paramVec = paramVec.reshape(9,1)

    beta = paramVec[0:6,:]
    sigma2 = paramVec[6,:][0]**2
    vecAE = paramVec[7:,:]**2

    e = Y - X @ beta


    # Inital D (dict version)
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = vecAE[0,0]*KinshipA[k] + vecAE[1,0]*KinshipD[k]

    # ------------------------------------------------------------------------------
    # Obtain (I+D)^{-1}
    # ------------------------------------------------------------------------------
    invIplusDdict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        invIplusDdict[k] = np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])


    # (D+I)^{-1} (matrix version)
    invIplusD = scipy.sparse.lil_matrix((n,n))
    counter = 0
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):

            # Add a block for each level of each factor.
            invIplusD[Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = invIplusDdict[k]
            counter = counter + 1

    # Update e'V^(-1)e
    etinvVe = e.transpose() @ invIplusD @ e

    # Work out log|V| using the fact V is block diagonal
    logdetV = 0
    for k in np.arange(r):
        logdetV = logdetV - nlevels[k]*np.log(np.linalg.det(invIplusDdict[k]))

    # Work out the log likelihood
    llhcurr = 0.5*(n*np.log(sigma2)+(1/sigma2)*etinvVe + logdetV)

    return(llhcurr)

print('ADE result')
paramVecADE = np.array(tmp2[0])
paramVecADE[7:,:] = paramVecADE[7:,:]*paramVecADE[6,0]
print(paramVecADE)
print(llh_ADE(tmp2[0], X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipC))
print('Done')

betaADE = paramVecADE[0:6,:]
sigma2ADE = paramVecADE[6,0]**2

DdictADE = dict()
for k in np.arange(len(nraneffs)):
    # Construct D using sigma^2A and sigma^2D
    DdictADE[k] = vecAE[0,0]**2*KinshipA[k] + vecAE[1,0]**2*KinshipD[k]

for i in np.arange(len(nraneffs)):
    for j in np.arange(nlevels[i]):
        # Add block
        if i == 0 and j == 0:
            DADE = DdictADE[i]
        else:
            DADE = scipy.linalg.block_diag(DADE, DdictADE[i])


DinvIplusZtZDADE = DADE @ np.linalg.pinv(np.eye(DADE.shape[0]) + DADE)

t1 = time.time()
tmp = minimize(llh_ADE, initParams, args=(X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipC), method='Nelder-Mead', tol=1e-10)
# tmp = minimize(llh_ADE, initParams, args=(X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipD), method='BFGS', tol=1e-3)
# tmp = minimize(llh_ADE, initParams, args=(X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipD), method='Newton-CG', tol=1e-3)
# tmp = minimize(llh_ADE, initParams, args=(X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipD), method='trust-ncg', tol=1e-3)
# tmp = minimize(llh_ADE, initParams, args=(X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipD), method='trust-krylov', tol=1e-3)
# tmp = minimize(llh_ADE, initParams, args=(X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipD), method='trust-exact', tol=1e-3)

t2 = time.time()
print(t2-t1)

print('Optimizer result')
paramVecOpt = tmp['x'].reshape(9,1)
paramVecOpt[7:,:] = paramVecOpt[7:,:]*paramVecOpt[6,0]
print(paramVecOpt)
print(np.array([[tmp['fun']]]))

t1 = time.time()
betaOLS = np.linalg.pinv(X.transpose() @ X) @ X.transpose() @ Y
e = Y - X @ betaOLS
sigmaOLS = np.sqrt(e.transpose() @ e/(n-p))
t2 = time.time()
print(t2-t1)

sigma2OLS = sigmaOLS**2

paramVecOLS = np.zeros((9,1))
paramVecOLS[0:6,:] = betaOLS
paramVecOLS[6,:] = sigmaOLS[0,0]

print('OLS result')
print(paramVecOLS)
print(llh_ADE(paramVecOLS, X, Y, n, nlevels, nraneffs, Dinds, KinshipA, KinshipC))

for i in np.arange(X.shape[1]):

    # Get beta
    L = np.zeros((1,X.shape[1]))
    L[0,i]=1

    TADE = get_T2D(L, XtX, X.transpose(), DinvIplusZtZDADE, betaADE, sigma2ADE)
    #get_T2D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2)
    df_ADE = get_swdf_T2D(L, DADE, sigma2ADE, XtX, X.transpose(), X, np.eye(n), n, nlevels, nraneffs)

    pvalADE = 10**(-T2P2D(TADE,df_ADE,minlog=-1e20))

    TOLS = (L @ betaOLS)/ np.sqrt(sigma2OLS*(L @ np.linalg.pinv(XtX) @ L.transpose()))

    df_OLS = n-p

    pvalOLS =10**(-T2P2D(TOLS,df_OLS,minlog=-1e20))

    print('T', TADE, TOLS)
    print('p', pvalADE,pvalOLS)
    print('df', df_ADE,df_OLS)
