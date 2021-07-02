#!/apps/well/R/3.4.3/bin/Rscript
#$ -cwd
#$ -q short.qc
#$ -o sim/sim$simInd/simlog/
#$ -e sim/sim$simInd/simlog/

library(MASS)
library(Matrix)
library(lme4)
library(lmerTest)
library(tictoc)
library(yaml)

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

# Read in the BLMM inputs
inputs <- yaml.load_file(paste(outDir,"/inputs.yml",sep=''))

# Read in the response vector
all_Y <- read.csv(file = paste(outDir,'/data/Y_Rversion_',toString(batchNo),'.csv',sep=''),sep=',', header=FALSE)


# Read in the fixed effects design
X <- read.csv(file = inputs$X,sep=',', header=FALSE)

# Read in the factor vector for the first random factor in the design
Zfactor0 <- factor(read.csv(file = inputs$Z[[1]]$f1$factor, sep=',', header=FALSE)[,1])

# Read in the raw regressor for the first random factor in the design
Zdata0 <- read.csv(file = inputs$Z[[1]]$f1$design,sep=',', header=FALSE)

# Number of voxels we have
nvox <- dim(all_Y)[2]

# Empty array for beta estimates
betas <- matrix(0,dim(all_Y)[2],4)

# Empty array for sigma2 estimates
sigma2 <- matrix(0,dim(all_Y)[2],1)

# Empty array for computation times
times <- matrix(0,dim(all_Y)[2],1)

# Empty array for T statistics
Tstats <- matrix(0,dim(all_Y)[2],1)

# Empty array for pvals
Pvals <- matrix(0,dim(all_Y)[2],1)


if (dim(Zdata0)[2]==2){
    # Empty array for vechD estimates
    vechD <- matrix(0,dim(all_Y)[2],3)
} else if (dim(Zdata0)[2]==1) {
    # Empty array for vechD estimates
    vechD <- matrix(0,dim(all_Y)[2],1)
}

# Empty array for log-likelihoods
llh <- matrix(0,dim(all_Y)[2],1)

# Loop through each model and run lmer for each voxel
for (i in 1:nvox){
  
  # Use lmerTest for everything but timing
  lmer <- lmerTest::lmer

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
    x5 <- as.matrix(X[,5])[y!=0]

    if (dim(Zdata0)[2]==2){

        # Reformat the raw regressor matrix for the first random factor into columns
        z01 <- as.matrix(Zdata0[,1])[y!=0]
        z02 <- as.matrix(Zdata0[,2])[y!=0]

        # Drop missing from factor 0
        Zf0 <- Zfactor0[y!=0]

    } else if (dim(Zdata0)[2]==1){

        # Reformat the raw regressor matrix for the first random factor into columns
        z01 <- as.matrix(Zdata0[,1])[y!=0]

        # Drop missing from factor 0
        Zf0 <- Zfactor0[y!=0]

    }

    # Finally, drop any missing Y
    y <- y[y!=0]

    if (dim(Zdata0)[2]==2){

        # Run the model
        m <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + x5 + (0 + z01 + z02|Zf0), REML=TRUE)
        
        # Timing function with lme4
        lmer <- lme4::lmer

        # Get the function which is optimized
        devfun <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + x5 + (0 + z01 + z02|Zf0), REML=TRUE, devFunOnly = TRUE)


    } else if (dim(Zdata0)[2]==1){

        # Run the model
        m <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + x5 + (0 + z01|Zf0), REML=TRUE)
        
        # Timing function with lme4
        lmer <- lme4::lmer

        # Get the function which is optimized
        devfun <- lmer(y ~ 0 + x1 + x2 + x3 + x4 + x5 + (0 + z01|Zf0), REML=TRUE, devFunOnly = TRUE)

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

    # Recover D parameters
    Ds <- as.matrix(Matrix::bdiag(VarCorr(m)))

    if (dim(Zdata0)[2]==2){

        # Record fixed effects variance estimate
        sigma2[i,1]<-as.data.frame(VarCorr(m))$vcov[4]

        # Calculate vech(D_0)*sigma2
        vechD0 <- Ds[1:2,1:2][lower.tri(Ds[1:2,1:2],diag = TRUE)]

        # Record vech(D_0)
        vechD[i,1:3]<-vechD0/as.data.frame(VarCorr(m))$vcov[4]

    } else if (dim(Zdata0)[2]==1){

        # Record fixed effects variance estimate
        sigma2[i,1]<-as.data.frame(VarCorr(m))$vcov[2]

        # Calculate vech(D_0)*sigma2
        vechD0 <- Ds[1:1,1:1][lower.tri(Ds[1:1,1:1],diag = TRUE)]

        # Record vech(D_0)
        vechD[i,1:1]<-vechD0/as.data.frame(VarCorr(m))$vcov[2]

    }

    # Record log likelihood
    llh[i,1] <- logLik(m, REML=TRUE)[1]
    
    # Run T statistic inference
    Tresults<-lmerTest::contest1D(m, c(0,0,0,1),ddf=c("Satterthwaite"))
    
    # Get the T statistic
    Tstat<-Tresults$`t value`
    
    # Get the P value
    p<-Tresults$`Pr(>|t|)`
    
    # Make p-values 1 sided
    if (Tstat>0){
      p <- p/2
    } else {
      p <- 1-p/2
    }
    
    # Record p value
    Pvals[i,1] <- p
    
    # Record T stat
    Tstats[i,1] <- Tstat
    
    
  }

}

# Directory for lmer results for this simulation
lmerDir <- file.path(outDir, 'lmer')

# Make directory if it doesn't exist already
if (!file.exists(lmerDir)) {
  dir.create(lmerDir)
}

print(lmerDir)

# Write results back to csv file
write.csv(betas,paste(lmerDir,'/beta_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(sigma2,paste(lmerDir,'/sigma2_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(vechD,paste(lmerDir,'/vechD_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(llh,paste(lmerDir,'/llh_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(times,paste(lmerDir,'/times_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(Tstats,paste(lmerDir,'/Tstat_',toString(batchNo),'.csv',sep=''), row.names = FALSE)
write.csv(Pvals,paste(lmerDir,'/Pval_',toString(batchNo),'.csv',sep=''), row.names = FALSE)

# Remove the R file for this batch as we no longer need it
file.remove(paste(outDir,'/data/Y_Rversion_',toString(batchNo),'.csv',sep=''))