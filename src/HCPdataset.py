import pandas as pd
import numpy as np
from lib.npMatrix2d import *
from src.ADE import *
from scipy.optimize import minimize
from scipy import stats

cov = 'ACE'
model = 23
if model in [5,12,13,14,17, 18, 19, 23]:
    needPMAT=True
else:
    needPMAT=False

if needPMAT==True:
    famTypewoDZ = pd.read_csv('/home/tommaullin/Documents/BLMM_creation/hcpFamTypeswithoutDZ.csv')
else:
    # Read in family type dataset
    famTypewoDZ = pd.read_csv('/home/tommaullin/Documents/BLMM_creation/hcpFamTypeswithoutDZ.csv')

print(famTypewoDZ.shape)

# Remove columns not of interest.
reducedData = famTypewoDZ[['Subject','familyType','familyID']].sort_values(by=['familyType','familyID'])

print(reducedData.shape)

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
if model == 23:
    reducedUnrestricted = unrestricted[['Subject','Gender','ReadEng_Unadj','PSQI_Score']]
if model == 22:
    reducedUnrestricted = unrestricted[['Subject','Gender','Language_Task_Acc','MMSE_Score']]
if model == 24:
    reducedUnrestricted = unrestricted[['Subject','Gender','Language_Task_Acc','PSQI_Score']]
if model == 25:
    reducedUnrestricted = unrestricted[['Subject','Gender','ReadEng_Unadj','MMSE_Score']]
if model == 26:
    reducedUnrestricted = unrestricted[['Subject','Gender','PSQI_Score','MMSE_Score']]

# Total Hippo
#reducedUnrestricted['FS_Total_Hippo_Vol'] = reducedUnrestricted['FS_L_Hippo_Vol'] + reducedUnrestricted['FS_R_Hippo_Vol']

# Demean and rescale FS_Total_GM_Vol, FS_Intracranial_Vol, FS_L_Hippo_Vol, FS_R_Hippo_Vol
#reducedUnrestricted.loc[:,'FS_Total_GM_Vol'] = (reducedUnrestricted['FS_Total_GM_Vol'] - reducedUnrestricted['FS_Total_GM_Vol'].mean())/reducedUnrestricted['FS_Total_GM_Vol'].std()
#reducedUnrestricted.loc[:,'FS_IntraCranial_Vol'] = (reducedUnrestricted['FS_IntraCranial_Vol'] - reducedUnrestricted['FS_IntraCranial_Vol'].mean())/reducedUnrestricted['FS_IntraCranial_Vol'].std()


# reducedUnrestricted.loc[:,'FS_Total_GM_Vol'] = reducedUnrestricted.loc[:,'FS_Total_GM_Vol']**(1/3)
# reducedUnrestricted.loc[:,'FS_IntraCranial_Vol'] = reducedUnrestricted.loc[:,'FS_IntraCranial_Vol']**(1/3)
# reducedUnrestricted.loc[:,'FS_L_Hippo_Vol'] = reducedUnrestricted.loc[:,'FS_L_Hippo_Vol']**(1/3)
# reducedUnrestricted.loc[:,'FS_R_Hippo_Vol'] = reducedUnrestricted.loc[:,'FS_R_Hippo_Vol']**(1/3)
# reducedUnrestricted.loc[:,'FS_Total_Hippo_Vol'] = reducedUnrestricted.loc[:,'FS_Total_Hippo_Vol']**(1/3)


# Add unrestricted into table and drop na values
newTab = pd.merge(newTab,reducedUnrestricted,on=['Subject']).dropna()

#.sort_values(by=['ZygosityGT'],ascending=False)
newTab['Gender']=newTab['Gender'].replace(['F','M'],[0,1])

# Add age and sex interaction
newTab['Age:Sex']=newTab[['Age_in_Yrs']].values*newTab[['Gender']].values

# Apply the appropriate sort
newTab=newTab.sort_values(by=['familyType','familyID','ParentCounts','ZygosityGT','ZygositySR'],ascending=[True,True,False,True,False])

# -----------------------------------------------------------------------------------
# Check families are coded correctly
# -----------------------------------------------------------------------------------

# Work out the unique types of family
UniqueFamilyTypes, idx = np.unique(newTab[['familyType']], return_index=True)
UniqueFamilyTypes = UniqueFamilyTypes[np.argsort(idx)]

# Number of grouping factors r
r = len(UniqueFamilyTypes)

# Loop through each family type (these are our factors)
for k in np.arange(r):

    # Work out which family type we're looking at
    uniqueType = UniqueFamilyTypes[k]

    # Get the table of these families
    familyTypeTable = newTab[newTab['familyType']==uniqueType]

    # Get a list of all family IDs in this category
    uniqueFamilyIDs = np.unique(familyTypeTable[['familyID']])

    # Loop through each family and work out the number of family members
    noFamMem = 0
    for j in np.arange(len(uniqueFamilyIDs)):

        # Get the ID for this family
        famID = uniqueFamilyIDs[j]

        # Get the table for this ID
        famTable = familyTypeTable[familyTypeTable['familyID']==famID]

        # Work out the number of subjects in this family
        noFamMem = np.maximum(noFamMem, famTable.shape[0])

    # Loop through each family and check they have all family members
    for j in np.arange(len(uniqueFamilyIDs)):

        # Get the ID for this family
        famID = uniqueFamilyIDs[j]

        # Get the table for this ID
        famTable = familyTypeTable[familyTypeTable['familyID']==famID]

        # If we don't have all subjects drop this family (we could recalculate
        # the family indexes... but this is only an illustrative example).
        if noFamMem > famTable.shape[0]:

            # Drop the familys that are now missing subjects.
            #newTab = newTab.drop(newTab[newTab.familyID == famID].index)

            newTab['familyType'][newTab.familyID == famID] = np.amax(UniqueFamilyTypes)+1
            UniqueFamilyTypes = np.append(UniqueFamilyTypes,np.amax(UniqueFamilyTypes)+1)

# Recalculate the unique types of family
UniqueFamilyTypes, idx = np.unique(newTab[['familyType']], return_index=True)
UniqueFamilyTypes = UniqueFamilyTypes[np.argsort(idx)]

# Recalculate number of grouping factors r
r = len(UniqueFamilyTypes)

# -----------------------------------------------------------------------------------
# Construct X
# -----------------------------------------------------------------------------------

if model in [1, 2, 18, 19, 20, 21]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','FS_IntraCranial_Vol','FS_Total_GM_Vol']].values 
elif model in [3, 6, 7, 8]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','FS_IntraCranial_Vol','ReadEng_AgeAdj']].values 
elif model in [4, 9, 10, 11]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','FS_IntraCranial_Vol','ReadEng_Unadj']].values 
elif model in [5, 12, 13, 14]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','FS_IntraCranial_Vol','PMAT24_A_CR']].values 
elif model in [15]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','FS_IntraCranial_Vol','FS_L_Hippo_Vol']].values
elif model in [16]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','FS_IntraCranial_Vol','FS_Total_GM_Vol','FS_L_Hippo_Vol']].values
elif model in [17, 23, 24, 26]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','PSQI_Score']].values 
elif model in [22, 25]:
    X = newTab[['Age_in_Yrs','Gender','Age:Sex','MMSE_Score']].values 

# Add an intercept to X
X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

# -----------------------------------------------------------------------------------
# Construct Y
# -----------------------------------------------------------------------------------

# Construct Y
if model in [1, 15, 16, 17]:
    Y = newTab[['ReadEng_AgeAdj']].values
elif model in [2, 23, 25]:
    Y = newTab[['ReadEng_Unadj']].values
elif model in [3, 4, 5]:
    Y = newTab[['FS_Total_GM_Vol']].values
elif model in [6, 9, 12]:
    Y = newTab[['FS_L_Hippo_Vol']].values
elif model in [7, 10, 13]:
    Y = newTab[['FS_R_Hippo_Vol']].values
elif model in [8, 11, 14]:
    Y = newTab[['FS_Total_Hippo_Vol']].values
elif model in [18, 19]:
    Y = newTab[['PMAT24_A_CR']].values
elif model in [20, 21, 26]:
    Y = newTab[['MMSE_Score']].values
elif model in [22, 24]:
    Y = newTab[['Language_Task_Acc']].values

# Number of fixed effects parameters p
p = X.shape[1]

# Output files for running same analysis in R
pd.DataFrame(X).to_csv('/home/tommaullin/Documents/BLMM_creation/X_HCP.csv')
pd.DataFrame(Y).to_csv('/home/tommaullin/Documents/BLMM_creation/Y_HCP.csv')


# -----------------------------------------------------------------------------------
# Calculate Kinship matrices
# -----------------------------------------------------------------------------------

# Number of levels and random effects for each factor
nlevels = np.zeros(r)
nraneffs = np.zeros(r)

# Dictionary to store Kinship matrices
KinshipA = dict()
KinshipC = dict()
KinshipD = dict()

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

# -----------------------------------------------------------------------------------
# Calculate Structure matrices of first kind
# -----------------------------------------------------------------------------------

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

    if cov == 'ADE':
        # Construct structure matrices
        structMat1stDict[k]=np.concatenate((SkrowA,SkrowD),axis=0)
    if cov == 'ACE':
        # Construct structure matrices
        structMat1stDict[k]=np.concatenate((SkrowA,SkrowC),axis=0)

# Work out structure matrix of the second kind
structMat2nd = np.concatenate((np.tile([[1,0]],r),np.tile([[0,1]],r)),axis=0)

# Convergence tolerance
tol = 1e-6

# Number of observations
n = X.shape[0]

# -----------------------------------------------------------------------------------
# Get the sum of X kron X (only for speeding up optimizer to ensure fair comparison)
# -----------------------------------------------------------------------------------

# Work out sum over j of X_(k,j) kron X_(k,j), for each k
XkXdict = dict()

# Loop through levels and factors
for k in np.arange(r):

    # Get qk
    qk = nraneffs[k]

    # Sum XkX
    XkXdict[k] = np.zeros((p**2,qk**2))

    for j in np.arange(nlevels[k]):

        # Indices for level j of factor k
        Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

        # Add to running sum
        XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

# -----------------------------------------------------------------------------------
# Run pFS
# -----------------------------------------------------------------------------------

reml=True

if cov == 'ADE':
    t1 = time.time()
    tmp2=pFS_ADE2D(X, Y, nlevels, nraneffs, tol, n, KinshipA, KinshipD, structMat1stDict, structMat2nd,reml=reml)
    t2 = time.time()
    print(t2-t1)
if cov == 'ACE':
    t1 = time.time()
    tmp2=pFS_ADE2D(X, Y, nlevels, nraneffs, tol, n, KinshipA, KinshipC, structMat1stDict, structMat2nd,reml=reml)
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



def llh_ADE(paramVec, X, Y, n, p, nlevels, nraneffs, Dinds, KinshipA, KinshipC, reml=False, XkXdict=None):

    paramVec = paramVec.reshape(p+3,1)

    beta = paramVec[0:p,:]
    sigma2 = paramVec[p,:][0]**2
    vecAE = paramVec[(p+1):,:]**2

    e = Y - X @ beta


    # Inital D (dict version)
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = forceSym2D(vecAE[0,0]*KinshipA[k] + vecAE[1,0]*KinshipC[k])

    # ------------------------------------------------------------------------------
    # Obtain (I+D)^{-1}
    # ------------------------------------------------------------------------------
    invIplusDdict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        invIplusDdict[k] = forceSym2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]))


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

    if reml:

        # Work out X'V^(-1)X as matrix reshape of (sum over k of ((sum_j X_(k,j) kron X_(k,j))vec(D_k)))
        XtinvVX = np.zeros((p,p))

        # Loop through levels and factors
        for k in np.arange(r):

            XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(invIplusDdict[k]),shape=np.array([p,p]))

        logdet = np.linalg.slogdet(XtinvVX)
        llhcurr = llhcurr - 0.5*logdet[0]*logdet[1] + 0.5*p*np.log(sigma2)

    return(llhcurr)

def varBeta_ADE(paramVec, p, KinshipA, KinshipC, nlevels, nraneffs):

    # Work out beta, sigma2 and the vector of variance components
    beta = paramVec[0:p,:]
    sigma2 = paramVec[p,0]**2
    vecAE = paramVec[(p+1):,:]**2/sigma2

    # Get D in dictionary form
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = vecAE[0,0]*KinshipA[k] + vecAE[1,0]*KinshipC[k]

    # r, total number of random factors
    r = len(nlevels)

    # Work out sum over j of X_(k,j) kron X_(k,j), for each k
    XkXdict = dict()

    # Loop through levels and factors
    for k in np.arange(r):

        # Get qk
        qk = nraneffs[k]

        # Sum XkX
        XkXdict[k] = np.zeros((p**2,qk**2))

        for j in np.arange(nlevels[k]):

            # Indices for level j of factor k
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Add to running sum
            XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

            
    # Work out X'V^(-1)X as matrix reshape of (sum over k of ((sum_j X_(k,j) kron X_(k,j))vec(D_k)))
    XtinvVX = np.zeros((p,p))

    # Loop through levels and factors
    for k in np.arange(r):

        XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])),shape=np.array([p,p]))

    # Check
    # Work out indices (there is one block of D per level)
    inds = np.zeros(np.sum(nlevels)+1)
    counter = 0
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            inds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1


    # Last index will be missing so add it
    inds[len(inds)-1]=inds[len(inds)-2]+nraneffs[-1]

    # Make sure indices are ints
    inds = np.int64(inds)

    # Initial D
    D = np.zeros((np.sum(nraneffs*nlevels),np.sum(nraneffs*nlevels)))

    counter = 0
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):

            D[inds[counter]:inds[counter+1], inds[counter]:inds[counter+1]] = Ddict[k]
            counter = counter + 1

    XtinvVX2 = X.transpose() @ np.linalg.inv(np.eye(D.shape[0])+D) @ X

    print('check: ', np.allclose(XtinvVX, XtinvVX2))


    # Get variance of beta
    varb = sigma2*np.linalg.inv(XtinvVX)

    print('s.e.(b): ', np.sqrt(np.diagonal(varb)))
    return(np.sqrt(np.diagonal(varb)))

print('ADE result')
paramVecADE = np.array(tmp2[0])
toDisplay = np.array(paramVecADE)
toDisplay[(p+1):,:] = toDisplay[(p+1):,:]*toDisplay[p,0]
print(toDisplay)
print(llh_ADE(paramVecADE, X, Y, n, p, nlevels, nraneffs, Dinds, KinshipA, KinshipC, reml=reml, XkXdict=XkXdict)-n/2*np.log(2*np.pi))
print('Done')

varBeta_ADE(toDisplay, p, KinshipA, KinshipC, nlevels, nraneffs)

for j in np.arange(p):
    L = np.zeros((1,p))
    L[0,j]=1
    
    swdf = get_swdf_ADE_T2D(L, paramVecADE, X, nlevels, nraneffs, KinshipA, KinshipC, structMat1stDict)
    print('swdf: ', swdf)
    T = get_T_ADE_2D(L, X, paramVecADE, KinshipA, KinshipC, nlevels, nraneffs)
    print('T: ', T)
    if T < 0:
        pvalADE = 1-stats.t.cdf(T, swdf)
    else:
        pvalADE = stats.t.cdf(-T, swdf)

    if pvalADE < 0.5:
        pvalADE = 2*pvalADE
    else:
        pvalADE = 2*(1-pvalADE)

    print('P: ', pvalADE)


t1 = time.time()
tmp = minimize(llh_ADE, initParams, args=(X, Y, n, p, nlevels, nraneffs, Dinds, KinshipA, KinshipC, reml, XkXdict), method='Nelder-Mead', tol=1e-6)
t2 = time.time()
print(t2-t1)

print('Optimizer result 1')
paramVecOpt = tmp['x'].reshape((p+3),1)
toDisplay = np.array(paramVecOpt)
toDisplay[(p+1):,:] = toDisplay[(p+1):,:]*toDisplay[p,0]
print(toDisplay)
print(np.array([[tmp['fun']]])-n/2*np.log(2*np.pi))

varBeta_ADE(toDisplay, p, KinshipA, KinshipC, nlevels, nraneffs)
for j in np.arange(p):
    L = np.zeros((1,p))
    L[0,j]=1
    
    swdf = get_swdf_ADE_T2D(L, paramVecOpt, X, nlevels, nraneffs, KinshipA, KinshipC, structMat1stDict)
    print('swdf: ', swdf)
    T = get_T_ADE_2D(L, X, paramVecOpt, KinshipA, KinshipC, nlevels, nraneffs)
    print('T: ', T)
    if T < 0:
        pvalOpt = 1-stats.t.cdf(T, swdf)
    else:
        pvalOpt = stats.t.cdf(-T, swdf)

    if pvalOpt < 0.5:
        pvalOpt = 2*pvalOpt
    else:
        pvalOpt = 2*(1-pvalOpt)

    print('P: ', pvalOpt)

t1 = time.time()
tmp = minimize(llh_ADE, initParams, args=(X, Y, n, p, nlevels, nraneffs, Dinds, KinshipA, KinshipC, reml, XkXdict), method='Powell', tol=1e-6)
t2 = time.time()
print(t2-t1)

print('Optimizer result 2')
paramVecOpt = tmp['x'].reshape((p+3),1)
toDisplay = np.array(paramVecOpt)
toDisplay[(p+1):,:] = toDisplay[(p+1):,:]*toDisplay[p,0]
print(toDisplay)
print(np.array([[tmp['fun']]])-n/2*np.log(2*np.pi))

varBeta_ADE(toDisplay, p, KinshipA, KinshipC, nlevels, nraneffs)
for j in np.arange(p):
    L = np.zeros((1,p))
    L[0,j]=1
    
    swdf = get_swdf_ADE_T2D(L, paramVecOpt, X, nlevels, nraneffs, KinshipA, KinshipC, structMat1stDict)
    print('swdf: ', swdf)
    T = get_T_ADE_2D(L, X, paramVecOpt, KinshipA, KinshipC, nlevels, nraneffs)
    print('T: ', T)
    if T < 0:
        pvalOpt = 1-stats.t.cdf(T, swdf)
    else:
        pvalOpt = stats.t.cdf(-T, swdf)

    if pvalOpt < 0.5:
        pvalOpt = 2*pvalOpt
    else:
        pvalOpt = 2*(1-pvalOpt)

    print('P: ', pvalOpt)


t1 = time.time()
betaOLS = np.linalg.pinv(X.transpose() @ X) @ X.transpose() @ Y
e = Y - X @ betaOLS
if not reml:
    sigmaOLS = np.sqrt(e.transpose() @ e/n)
else:
    sigmaOLS = np.sqrt(e.transpose() @ e/(n-p))
t2 = time.time()
print(t2-t1)

sigma2OLS = sigmaOLS**2

paramVecOLS = np.zeros(((p+3),1))
paramVecOLS[0:p,:] = betaOLS
paramVecOLS[p,:] = sigmaOLS[0,0]
toDisplay = np.array(paramVecOLS)

print('OLS result')
print(toDisplay)
print(llh_ADE(toDisplay, X, Y, n, p, nlevels, nraneffs, Dinds, KinshipA, KinshipC, reml=reml, XkXdict=XkXdict)-n/2*np.log(2*np.pi))
varBeta_ADE(toDisplay, p, KinshipA, KinshipC, nlevels, nraneffs)

for i in np.arange(X.shape[1]):

    # Get beta
    L = np.zeros((1,X.shape[1]))
    L[0,i]=1

    TOLS = (L @ betaOLS)/ np.sqrt(sigma2OLS*(L @ np.linalg.pinv(XtX) @ L.transpose()))

    df_OLS = n-p

    if T < 0:
        pvalOLS = 1-stats.t.cdf(TOLS, df_OLS)
    else:
        pvalOLS = stats.t.cdf(-TOLS, df_OLS)

    if pvalOLS < 0.5:
        pvalOLS = 2*pvalOLS
    else:
        pvalOLS = 2*(1-pvalOLS)

    print('df: ', df_OLS)
    print('T: ', TOLS)
    print('pval: ', pvalOLS)
