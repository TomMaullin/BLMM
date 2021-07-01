#!/bin/bash

# Get the data in a nice format for R
fslpython -c "from lmer import Rpreproc; Rpreproc.Rpreproc('$2', $3, $4)"

# Load R
module load R/3.4.3 

# Run parameter estimation
R CMD BATCH --no-save --no-restore '--args batchNo='$4' outDir="'$2'" ' $1/lmer/lmer_paramEst.R $1/lmerlog/Rout'_'$4.txt

# Cleanup the files created
fslpython -c "from lmer import cleanup; cleanup.Rcleanup('$2', $3, $4)"