#!/bin/bash

# Load R
module load R/3.4.3 

echo $1
echo $2
echo $3

echo $3/sim$1/logDataGen/Rout.txt

# Run script
R CMD BATCH --no-save --no-restore '--args simInd=$1 batchNo=$2 outDir="$3"' $3/lmer_paramEst.R $3/sim$1/logDataGen/Rout.txt &