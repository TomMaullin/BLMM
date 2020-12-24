#!/bin/bash

# Load R
module load R/3.4.3 

# Run script
R CMD BATCH --no-save --no-restore '--args simInd=$1 batchNo=$2 outDir="$3"' $SIM_PATH/lmer_paramEst.R $SIM_PATH/sim$simInd/logDataGen/Rout.txt &