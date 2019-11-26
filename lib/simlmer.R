#!/apps/well/R/3.4.3/bin/Rscript
#$ -cwd
#$ -q short.qc
#$ -o ./loglmer/
#$ -e ./loglmer/

library(MASS)
library(Matrix)
#install.packages("lme4")
library(lme4, lib.loc="/users/nichols/inf852/BLMM/Rpackages/")
#install.packages("tictoc")
library(tictoc, lib.loc="/users/nichols/inf852/BLMM/Rpackages/")

runningtime <- 0
runningbdiff <- 0
runningbetadiff <- 0
for (i in seq(20)){
  for (j in seq(20)){
    for (k in seq(20)){
          
    # Say 30 subs, 34ish readings each
    fS <- gl(30, 34)[1:1000]#gl(20, 50)
    JS <- t(as(fS, Class = "sparseMatrix"))[1:1000,1:30]
    num_s <- 5
    
    # Random intercept and random readings as RFX
    XS <- cbind(rnorm(1000),rnorm(1000),rnorm(1000))
    
    # Generate random effects matrix for subject factor
    ZS <- t(KhatriRao(t(JS), t(XS)))
    
    # Say 3 sites/groups, subjects randomly scanned at
    fG <- as.factor(sample(c(1:10),1000,replace=TRUE))
    JG <- t(as(fG, Class = "sparseMatrix"))
    
    # Random intercept and random readings as RFX
    XG <- rnorm(1000)#rep(1,1000)#cbind(1,rnorm(1000))
    num_g <- 2
    
    # Generate random effects matrix for group factor
    ZG <- t(KhatriRao(t(JG), t(XG)))
    
    # Construct RFX matrix
    Z <- cbind(ZS, ZG)
    
    # Image of Z'Z
    image(t(Z)%*%Z)
    
    # Image of Zi'Zi where Z1 is the first 10 rows of Z, 
    # z2 is second and so on
    #image(t(Z[1:10,])%*%Z[1:10,])
    #image(t(Z[11:20,])%*%Z[11:20,])
    
    # Fixed effects matrix
    X <- cbind(1, rnorm(1000), rnorm(1000), rnorm(1000), rnorm(1000))
    
    # Generate b
    b <- 20*rnorm(100)
      
    # Generate beta
    beta <- 20*rnorm(5)
    
    # Generate response 
    y <- X%*%beta + Z%*%b + rnorm(1000)
    
    #------------------------------------------------------------------------------------------------
    # Now to see if lmer can decipher this model
    y <- as.matrix(y)
    
    x1 <- as.matrix(X[,1])
    x2 <- as.matrix(X[,2])
    x3 <- as.matrix(X[,3])
    x4 <- as.matrix(X[,4])
    x5 <- as.matrix(X[,5])
    
    z1 <- as.matrix(XS[,1])
    z2 <- as.matrix(XS[,2])
    z3 <- as.matrix(XS[,3])
    #z4 <- as.matrix(XG[,1])# CAREFUL HERE
    
    tic('lmer time')
    m <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z1 + z2 + z3|fS) + (0 + XG|fG)) #Don't need intercepts in R - automatically assumed
    t<-toc()
    
    runningtime <- runningtime + t$toc - t$tic
    
    # RFX variances 
    as.matrix(Matrix::bdiag(VarCorr(m)))
    
    betahat <- fixef(m)
    
    
    tmp1 <- t(matrix(data=b[1:90],nrow=3,ncol=30))
    tmp1_est <- ranef(m)$fS
    diff1<-as.list(abs(tmp1-tmp1_est))
    
    
    tmp2 <- t(matrix(data=b[91:100],nrow=1,ncol=10))
    tmp2_est <- ranef(m)$fG
    diff2<-as.list(abs(tmp2-tmp2_est))
    
    runningbdiff <- runningbdiff + mean(rbind(diff1$z2, diff1$z3, diff1$`(Intercept)`, diff2$XG))
    runningbetadiff <- runningbetadiff + mean(abs(beta-betahat))
    }}}

print(runningtime)
print(runningbdiff/(20*20*20))
print(runningbetadiff/(20*20*20))
