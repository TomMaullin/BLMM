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
# Read and parse BLMM config file
# -----------------------------------------------------------------------
# Work out if we have been given multiple analyses configurations
# Else just assume blmm_config is the correct configuration
if [ "$1" == "" ] ; then
  cfg=$(RealPath "blmm_config.yml")
else
  cfg=$1
fi
cfg=$(RealPath $cfg)

# read yaml file to get output directory
eval $(parse_yaml $cfg "config_")

# -----------------------------------------------------------------------
# Make output directory and empty nb.txt file
# -----------------------------------------------------------------------
mkdir -p $config_outdir

# This file is used to record number of batches
if [ -f $config_outdir/nb.txt ] ; then
    rm $config_outdir/nb.txt 
fi
touch $config_outdir/nb.txt 

# -----------------------------------------------------------------------
# Make a copy of the inputs file, if the user touches the cfg file this
# will not mess with anything now.
# -----------------------------------------------------------------------
inputs=$config_outdir/inputs.yml
cp $cfg $inputs

# -----------------------------------------------------------------------
# Submit setup job
# -----------------------------------------------------------------------
fsl_sub -l log/ -N setup bash $BLMM_PATH/scripts/cluster_blmm_setup.sh $inputs > /tmp/$$ && setupID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$setupID" == "" ] ; then
  echo "Setup job submission failed!"
else
  echo "Submitted: Setup job."
fi

# This loop waits for the setup job to finish before
# deciding how many batches to run. It also checks to 
# see if the setup job has errored.
nb=0
i=0
while [ $nb -lt 1 ]
do

  # obtain number of batches
  sleep 1
  if [ -s $config_outdir/nb.txt ]; then
    typeset -i nb=$(cat $config_outdir/nb.txt)
  fi
  i=$(($i + 1))

  # Check for error
  if [ $i -gt 30 ]; then
    errorlog="log/setup.e$setupID"
    if [ -s $errorlog ]; then
      echo "Setup has errored"
      exit
    fi
  fi

  # Timeout
  if [ $i -gt 500 ]; then
    echo "Something seems to be taking a while. Please check for errors."
  fi
done

# Reread yaml file in case filepaths have been updated to be absolute
eval $(parse_yaml $inputs "config_")
inputs=$config_outdir/inputs.yml

# -----------------------------------------------------------------------
# Submit Batch jobs
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
fsl_sub -j $batchIDs -l log/ -N concat bash $BLMM_PATH/scripts/cluster_blmm_concat.sh $inputs > /tmp/$$ && concatID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$concatID" == "" ] ; then
  echo "Concatenation job submission failed!"
else
  echo "Submitted: Concatenation job."
fi

# -----------------------------------------------------------------------
# Submit Results jobs
# -----------------------------------------------------------------------
# Check if we are in voxel batch mode (not yet implemented)
if [ -z $config_voxelBatching ] || [ "$config_voxelBatching" == "0" ] ; then
    
  # Voxel batching is not turned on
  fsl_sub -j $concatID -l log/ -N results bash $BLMM_PATH/scripts/cluster_blmm_results.sh $inputs "-1" > /tmp/$$ && resultsIDs=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
  if [ "$resultsIDs" == "" ] ; then
    echo "Results job submission failed!"
  else
    echo "Submitted: Results job."
  fi

else

  # Voxel batching is on, so work out number of voxel batches needed.
  typeset -i nvb=$(cat $config_outdir/nvb.txt)

  i=1
  while [ "$i" -le "$nvb" ]; do

    # Submit nb batches and get the ids for them
    fsl_sub -j $concatID -l log/ -N results$i bash $BLMM_PATH/scripts/cluster_blmm_results.sh $inputs $i > /tmp/$$ && resultsIDs=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$),$resultsIDs
    i=$(($i + 1))
  done
  echo "Submitted: Results jobs (in Voxel Batch mode)."

fi

# -----------------------------------------------------------------------
# Submit Cleanup job
# -----------------------------------------------------------------------
fsl_sub -j $resultsIDs -l log/ -N cleanup bash $BLMM_PATH/scripts/cluster_blmm_cleanup.sh $inputs > /tmp/$$ && cleanupID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$cleanupID" == "" ] ; then
  echo "Clean up job submission failed!"
else
  echo "Submitted: Cleanup job."
fi
echo "Analysis submission complete. Please use qstat to monitor progress."