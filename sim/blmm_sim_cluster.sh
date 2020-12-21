#!/bin/bash

echo "Setting up distributed analysis..."

# -----------------------------------------------------------------------
# Work out BLMM path
# -----------------------------------------------------------------------
RealPath() {
    (echo $(cd $(dirname "$1") && pwd -P)/$(basename "$1"))
}

BLMM_PATH=$(dirname $(RealPath "${BASH_SOURCE[0]}"))

# include parse_yaml function
. $BLMM_PATH/scripts/parse_yaml.sh

# -----------------------------------------------------------------------
# Submit data generation job
# -----------------------------------------------------------------------
fsl_sub -l sim/sim$simInd/logDataGen/ -N dataGen$simInd bash $BLMM_PATH/sim/generate_data.sh $i $inputs > /tmp/$$ && dataGenID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)

# This loop waits for the data generation job to finish before
# deciding how many batches to run. It also checks to see if the data
# generation job has errored.
nb=0
i=0
while [ $nb -lt 1 ]
do

  # obtain number of batches
  sleep 1
  if [ -s $BLMM_PATH/sim/sim$simInd/data/nb.txt ]; then
    typeset -i nb=$(cat $BLMM_PATH/sim/sim$simInd/data/nb.txt)
  fi
  i=$(($i + 1))

  # Check for error
  if [ $i -gt 30 ]; then
    errorlog="/sim/sim$simInd/logDataGen/dataGen.e$setupID"
    if [ -s $errorlog ]; then
      echo "Data generation has errored"
      exit
    fi
  fi

  # Timeout
  if [ $i -gt 1000 ]; then
    echo "Something seems to be taking a while. Please check for errors."
  fi
done

# -----------------------------------------------------------------------
# Submit 
# -----------------------------------------------------------------------
i=1
while [ "$i" -le "$nb" ]; do

  # Submit nb batches and get the ids for them
  fsl_sub -j $setupID -l log/ -N batch${i} bash $BLMM_PATH/scripts/cluster_blmm_batch.sh $i $inputs > /tmp/$$ && batchIDs=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$),$batchIDs
  i=$(($i + 1))
done
if [ "$batchIDs" == "" ] ; then
  echo "Batch jobs submission failed!"
else
  echo "Submitted: Batch jobs."
fi

# -----------------------------------------------------------------------
# Submit Concatenation job
# -----------------------------------------------------------------------
# fsl_sub -j $batchIDs -l log/ -N concat bash $BLMM_PATH/scripts/cluster_blmm_concat.sh $inputs > /tmp/$$ && concatID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
# if [ "$concatID" == "" ] ; then
#   echo "Concatenation job submission failed!"
# else
#   echo "Submitted: Concatenation job."
# fi

