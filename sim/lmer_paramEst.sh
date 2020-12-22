#!/bin/bash
#$ -cwd
#$ -q short.qc
#$ -o sim/sim$simInd/logDataGen/
#$ -e sim/sim$simInd/logDataGen/

module load R/3.4.3 
R CMD BATCH $SIM_PATH/lmer_paramEst.R $1 $2 $3