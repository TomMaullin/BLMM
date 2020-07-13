library(haven)
library(lme4)
library(tictoc)

# Read in reduced dataset
reducedat <- as.data.frame(read.csv('/home/tommaullin/Documents/BLMM_creation/BLMM/sim/schools_reduced.csv'))

# Work out response and factors
y <- as.matrix(reducedat$math)
tchrfac <- as.factor(reducedat$tchrid)
studfac <- as.factor(reducedat$studid)
m1
# Work out design
x1 <- as.matrix(reducedat$year) # x0 intercept

# Run model
m1 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac),REML=FALSE)
devfun1 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac),REML=FALSE,devFunOnly = TRUE)

tic('lmer time')
tmp <- optimizeLmer(devfun1)
t <- toc()

# Fixed effects and fixed effects variances
fixef(m1)
summary(m1)$coef[, 2, drop = FALSE]

# RFX variances and residual variance
as.matrix(Matrix::bdiag(VarCorr(m1)))
resid <- as.data.frame(VarCorr(m1))[3,4]#as.data.frame(vc,order="lower.tri")

# Log-likelihood minus pi term
logLik(m1)+234/2*log(2*pi)