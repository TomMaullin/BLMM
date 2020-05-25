library(MASS)
library(Matrix)
library(lme4)
library(tictoc)

#trace(name_of_function, edit = T)

desInd <- 3
simInd <- 4

dataDir <- '/well/nichols/users/inf852/PaperSims'
if (desInd==3){
  
  results <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_results.csv',sep=''))
  
  X <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_X.csv',sep=''),sep=' ', header=FALSE)
  Y <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_Y.csv',sep=''),sep=' ', header=FALSE)
  Zfactor0 <- factor(read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_Zfactor0.csv',sep=''), sep=' ', header=FALSE)[,1])
  Zdata0 <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_Zdata0.csv',sep=''),sep=' ', header=FALSE)
  Zfactor1 <- factor(read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_Zfactor1.csv',sep=''), sep=' ', header=FALSE)[,1])
  Zdata1 <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_Zdata1.csv',sep=''),sep=' ', header=FALSE)
  Zfactor2 <- factor(read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_Zfactor2.csv',sep=''), sep=' ', header=FALSE)[,1])
  Zdata2 <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design3_Zdata2.csv',sep=''),sep=' ', header=FALSE)
  y <- as.matrix(Y[,1])
  
  x2 <- as.matrix(X[,2])
  x3 <- as.matrix(X[,3])
  x4 <- as.matrix(X[,4])
  x5 <- as.matrix(X[,5])
  
  
  z01 <- as.matrix(Zdata0[,1])
  z02 <- as.matrix(Zdata0[,2])
  z03 <- as.matrix(Zdata0[,3])
  z04 <- as.matrix(Zdata0[,4])
  
  z11 <- as.matrix(Zdata1[,1])
  z12 <- as.matrix(Zdata1[,2])
  z13 <- as.matrix(Zdata1[,3])
  
  z21 <- as.matrix(Zdata2[,1])
  z22 <- as.matrix(Zdata2[,2])
  
  tic('lmer time')
  m <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03 + z04|Zfactor0) + (0 + z11 + z12 + z13|Zfactor1) + (0 + z21 + z22|Zfactor2), REML = FALSE) #Don't need intercepts in R - automatically assumed
  t<-toc()
  
  lmertime <- t$toc-t$tic
  
  devfun <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03 + z04|Zfactor0) + (0 + z11 + z12 + z13|Zfactor1) + (0 + z21 + z22|Zfactor2), REML = FALSE, devFunOnly = TRUE) #Don't need intercepts in R - automatically assumed
  
  tic('lmer time 2')
  optimizeLmer(devfun)
  t<-toc()
  
  lmertime2 <- t$toc-t$tic
  
  
  results[1,'lmer']<-lmertime2
  results[3,'lmer']<-logLik(m)[1]
  results[4:8,'lmer'] <- fixef(m)
  
  Ds <- as.matrix(Matrix::bdiag(VarCorr(m)))
  
  vechD0 <- Ds[1:4,1:4][lower.tri(Ds[1:4,1:4],diag = TRUE)]
  vechD1 <- Ds[5:7,5:7][lower.tri(Ds[5:7,5:7],diag = TRUE)]
  vechD2 <- Ds[8:9,8:9][lower.tri(Ds[8:9,8:9],diag = TRUE)]
  
  results[29:38,'lmer']<-vechD0
  results[39:44,'lmer']<-vechD1
  results[45:47,'lmer']<-vechD2
  
  write.csv(results,paste(dataDir,'/Sim',toString(simInd),'_Design3_results.csv',sep=''), row.names = FALSE)
  
  
  
} else if (desInd==2){

  results <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design2_results.csv',sep=''))
  
  X <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design2_X.csv',sep=''),sep=' ', header=FALSE)
  Y <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design2_Y.csv',sep=''),sep=' ', header=FALSE)
  Zfactor0 <- factor(read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design2_Zfactor0.csv',sep=''), sep=' ', header=FALSE)[,1])
  Zdata0 <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design2_Zdata0.csv',sep=''),sep=' ', header=FALSE)
  Zfactor1 <- factor(read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design2_Zfactor1.csv',sep=''), sep=' ', header=FALSE)[,1])
  Zdata1 <- read.csv(file = paste(dataDir,'/Sim',toString(simInd),'_Design2_Zdata1.csv',sep=''),sep=' ', header=FALSE)
  
  y <- as.matrix(Y[,1])
  
  x2 <- as.matrix(X[,2])
  x3 <- as.matrix(X[,3])
  x4 <- as.matrix(X[,4])
  x5 <- as.matrix(X[,5])
  
  
  z01 <- as.matrix(Zdata0[,1])
  z02 <- as.matrix(Zdata0[,2])
  z03 <- as.matrix(Zdata0[,3])
  
  z11 <- as.matrix(Zdata1[,1])
  z12 <- as.matrix(Zdata1[,2])
  
  tic('lmer time')
  m <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03|Zfactor0) + (0 + z11 + z12|Zfactor1), REML = FALSE) #Don't need intercepts in R - automatically assumed
  t<-toc()
  
  lmertime <- t$toc-t$tic
  
  
  devfun <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03|Zfactor0) + (0 + z11 + z12|Zfactor1), REML = FALSE, devFunOnly = TRUE) #Don't need intercepts in R - automatically assumed
  
  tic('lmer time 2')
  optimizeLmer(devfun)
  t<-toc()
  
  lmertime2 <- t$toc-t$tic
  
  results[1,'lmer']<-lmertime
  results[3,'lmer']<-logLik(m)[1]
  results[4:8,'lmer'] <- fixef(m)
  
  Ds <- as.matrix(Matrix::bdiag(VarCorr(m)))
  
  vechD0 <- Ds[1:3,1:3][lower.tri(Ds[1:3,1:3],diag = TRUE)]
  vechD1 <- Ds[4:5,4:5][lower.tri(Ds[4:5,4:5],diag = TRUE)]
  
  results[19:24,'lmer']<-vechD0
  results[25:27,'lmer']<-vechD1
  
  write.csv(results,paste(dataDir,'/Sim',toString(simInd),'_Design2_results.csv',sep=''), row.names = FALSE)
  
}
results