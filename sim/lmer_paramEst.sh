#!/bin/bash

# Get the data in a nice format for R
fslpython -c "from sim import generateData; generateData.Rpreproc('$1', $2, [100,100,100], $3, $4)"

# Load R
module load R/3.4.3 

# Run parameter estimation
R CMD BATCH --no-save --no-restore '--args simInd='$2' batchNo='$4' outDir="'$1'"' $1/lmer_paramEst.R $1/sim$2/simlog/Rout$2'_'$4.txt

# Cleanup the files created
fslpython -c "from sim import cleanup; cleanup.Rcleanup('$1', $2, $3, $4)"
