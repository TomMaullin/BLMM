#!/bin/bash

# Get the data in a nice format for R
fslpython -c "from lmer import Rpreproc; Rpreproc.Rpreproc('$1', $2, $3)"

# Load R
module load R/3.4.3 

# Run parameter estimation
R CMD BATCH --no-save --no-restore '--args batchNo='$3' outDir="'$1'" ' $1/lmer_paramEst.R $1/lmerlog/Rout'_'$3.txt

# Cleanup the files created
fslpython -c "from lmer import cleanup; cleanup.Rcleanup('$1', $2, $3)"