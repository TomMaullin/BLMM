#!/apps/well/R/3.4.3/bin/Rscript
#$ -cwd
#$ -q short.qc
#$ -o sim/sim$simInd/simlog/
#$ -e sim/sim$simInd/simlog/

library(MASS)
library(Matrix)
library(lme4)
library(tictoc)

# ---------------------------------------------------------------------------------------
# IMPORTANT: Input options
# ---------------------------------------------------------------------------------------
#
# The below variables control which simulation is run and how. The variable names match
# those used in the `LMMPaperSim.py` file and are given as follows:
#
# - outDir: The output directory.
# - simInd: Simulation number.
# - batchNo: Batch number.
# 
# ---------------------------------------------------------------------------------------

print('heeeeere')
# Read in arguments from command line
args=(commandArgs(TRUE))

# Evaluate arguments
for(i in 1:length(args)){
  eval(parse(text=args[[i]]))
}

print(simInd)
print(batchNo)
print(outDir)
print(desInd)

#simInd <-20
#batchNo <- 51
#outDir <- '/home/tommaullin/Documents/BLMM/sim'

# Read in the fixed effects design
X <- read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/X.csv',sep=''),sep=',', header=FALSE)

# Read in the response vector
all_Y <- read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/Y_Rversion_',toString(batchNo),'.csv',sep=''),sep=',', header=FALSE)

# Read in the factor vector for the first random factor in the design
Zfactor0 <- factor(read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/f0.csv',sep=''), sep=',', header=FALSE)[,1])

# Read in the raw regressor for the first random factor in the design
Zdata0 <- read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/rr0.csv',sep=''),sep=',', header=FALSE)

if (desInd==3){
    # Read in the factor vector for the second random factor in the design
    Zfactor1 <- factor(read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/f1.csv',sep=''), sep=',', header=FALSE)[,1])

    # Read in the raw regressor for the second random factor in the design
    Zdata1 <- read.csv(file = paste(outDir,'/sim',toString(simInd),'/data/rr1.csv',sep=''),sep=',', header=FALSE)

}

# Number of voxels we have
nvox <- dim(all_Y)[2]

# Empty array for beta estimates
betas <- matrix(0,dim(all_Y)[2],4)

# Empty array for sigma2 estimates
sigma2 <- matrix(0,dim(all_Y)[2],1)

# Empty array for computation times
times <- matrix(0,dim(all_Y)[2],1)

if (desInd==3) {
    # Empty array for vechD estimates
    vechD <- matrix(0,dim(all_Y)[2],4)
} else if (desInd==2) {
    # Empty array for vechD estimates
    vechD <- matrix(0,dim(all_Y)[2],3)
} else if (desInd==1) {
    # Empty array for vechD estimates
    vechD <- matrix(0,dim(all_Y)[2],1)
}


# Empty array for log-likelihoods
llh <- matrix(0,dim(all_Y)[2],1)

# Loop through each model and run lmer for each voxel
for (i in 1:nvox){
  
  # Print i
  print(i)
  
  # Get Y
  y <- as.matrix(all_Y[,i])
  
  # If all y are zero this voxel was dropped from analysis as a 
  # result of missing data
  if (!all(y==0)){
    
    # Reformat X into columns and mask
    x1 <- as.matrix(X[,1])[y!=0]
    x2 <- as.matrix(X[,2])[y!=0]
    x3 <- as.matrix(X[,3])[y!=0]
    x4 <- as.matrix(X[,4])[y!=0]

    if (desInd==3){
    
        # Reformat the raw regressor matrix for the first random factor into columns
        z01 <- as.matrix(Zdata0[,1])[y!=0]
        z02 <- as.matrix(Zdata0[,2])[y!=0]
        
        # Reformat the raw regressor matrix for the second random factor into columns
        z11 <- as.matrix(Zdata1[,1])[y!=0]
        
        # Drop missing from factor 0
        Zf0 <- Zfactor0[y!=0]
        
        # Drop missing from factor 1
        Zf1 <- Zfactor1[y!=0]

    } else if (desInd==2){
    
        # Reformat the raw regressor matrix for the first random factor into columns
        z01 <- as.matrix(Zdata0[,1])[y!=0]
        z02 <- as.matrix(Zdata0[,2])[y!=0]
        
        # Drop missing from factor 0
        Zf0 <- Zfactor0[y!=0]

    } else if (desInd==1){

        # Reformat the raw regressor matrix for the first random factor into columns
        z01 <- as.matrix(Zdata0[,1])[y!=0]
        
        # Drop missing from factor 0
        Zf0 <- Zfactor0[y!=0]

    }
    
    # Finally, drop any missing Y
    y <- y[y!=0]
  
    if (desInd==3){

        # Run the model
        m <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + (0 + z01 + z02|Zf0) + (0 + z11|Zf1), REML=FALSE) 
      
        # Get the function which is optimized
        devfun <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + (0 + z01 + z02|Zf0) + (0 + z11|Zf1), REML=FALSE, devFunOnly = TRUE)

    } else if (desInd==2){

        # Run the model
        m <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + (0 + z01 + z02|Zf0), REML=FALSE) 
      
        # Get the function which is optimized
        devfun <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + (0 + z01 + z02|Zf0), REML=FALSE, devFunOnly = TRUE)


    } else if (desInd==1){

        # Run the model
        m <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + (0 + z01|Zf0), REML=FALSE) 
      
        # Get the function which is optimized
        devfun <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + (0 + z01|Zf0), REML=FALSE, devFunOnly = TRUE)

    }


    # Time lmer from only the devfun point onwards (to ensure fair comparison)
    tic('lmer time')
    opt<-optimizeLmer(devfun)
    t<-toc()
    
    # Calculate time
    lmertime <- t$toc-t$tic
    
    # Add to computation time
    times[i,1]<-lmertime
    
    # Record fixed effects estimates
    betas[i,1:4] <- fixef(m)
    
    # Record fixed effects variance estimate
    sigma2[i,1]<-as.data.frame(VarCorr(m))$vcov[5]
    
    # Recover D parameters
    Ds <- as.matrix(Matrix::bdiag(VarCorr(m)))
    
    if (desInd==3){

        # Calculate vech(D_0)*sigma2
        vechD0 <- Ds[1:2,1:2][lower.tri(Ds[1:2,1:2],diag = TRUE)]
        
        # Calculate vech(D_1)*sigma2
        vechD1 <- Ds[3:3,3:3][lower.tri(Ds[3,3],diag = TRUE)]
        
        # Record vech(D_0)
        vechD[i,1:3]<-vechD0/as.data.frame(VarCorr(m))$vcov[5]
        
        # Record vech(D_1)
        vechD[i,4:4]<-vechD1/as.data.frame(VarCorr(m))$vcov[5]

    } else if (desInd==2){

        # Calculate vech(D_0)*sigma2
        vechD0 <- Ds[1:2,1:2][lower.tri(Ds[1:2,1:2],diag = TRUE)]

        # Record vech(D_0)
        vechD[i,1:3]<-vechD0/as.data.frame(VarCorr(m))$vcov[5]        

    } else if (desInd==1){

        # Calculate vech(D_0)*sigma2
        vechD0 <- Ds[1:1,1:1][lower.tri(Ds[1:1,1:1],diag = TRUE)]

        # Record vech(D_0)
        vechD[i,1:1]<-vechD0/as.data.frame(VarCorr(m))$vcov[5]     

    }

    # Record log likelihood
    llh[i,1] <- logLik(m)[1]
    
  }
  
}


# Directory for lmer results for this simulation
lmerDir <- file.path(outDir, paste('sim',toString(simInd),sep=''),'lmer')

# Make directory if it doesn't exist already
if (!file.exists(lmerDir)) {
  dir.create(lmerDir)
}

# Write results back to csv file
write.csv(betas,paste(lmerDir,'/beta_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(sigma2,paste(lmerDir,'/sigma2_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(vechD,paste(lmerDir,'/vechD_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(llh,paste(lmerDir,'/llh_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(times,paste(lmerDir,'/times_',toString(batchNo),'.csv',sep=''), row.names = FALSE)

# Remove the R file for this batch as we no longer need it
file.remove(paste(outDir,'/sim',toString(simInd),'/data/Y_Rversion_',toString(batchNo),'.csv',sep=''))