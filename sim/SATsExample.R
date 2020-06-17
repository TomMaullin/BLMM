library(haven)
library(lme4)
library(tictoc)

# Read in reduced dataset
reducedat <- as.data.frame(read.csv('/home/tommaullin/Documents/BLMM_creation/BLMM/sim/schools_reduced.csv'))

# Work out response and factors
y <- as.matrix(reducedat$math)
tchrfac <- as.factor(reducedat$tchrid)
studfac <- as.factor(reducedat$studid)

# Work out design
x1 <- as.matrix(reducedat$year) # x0 intercept

# Run model
m1 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac),REML=FALSE)
devfun1 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac),REML=FALSE,devFunOnly = TRUE)

tic('lmer time')
tmp <- optimizeLmer(devfun1)
t <- toc()

# Fixed effects and fixed effects variances
fixef(m)
summary(m)$coef[, 2, drop = FALSE]

# RFX variances and residual variance
as.matrix(Matrix::bdiag(VarCorr(m)))
resid <- as.data.frame(VarCorr(m))[3,4]#as.data.frame(vc,order="lower.tri")

# Read in full dataset
fulldat <- as.data.frame(read.csv('/home/tommaullin/Documents/BLMM_creation/BLMM/sim/schools_full.csv'))

# Work out responses and factors
y <- as.matrix(fulldat$math)
tchrfac <- as.factor(fulldat$tchrid)
studfac <- as.factor(fulldat$studid)
schlfac <- as.factor(fulldat$schlid)

# Work out design
x1 <- as.matrix(fulldat$year) # x0 intercept

# Run model
m2 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac) + (1|schlfac),REML=FALSE)

devfun2 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac) + (1|schlfac),REML=FALSE,devFunOnly = TRUE)

tic('lmer time')
tmp <- optimizeLmer(devfun2)
t <- toc()