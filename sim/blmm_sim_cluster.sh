#!/bin/bash

echo "Setting up simulation..."

# -----------------------------------------------------------------------
# Work out BLMM path
# -----------------------------------------------------------------------
RealPath() {
    (echo $(cd $(dirname "$1") && pwd -P)/$(basename "$1"))
}

SIM_PATH=$(dirname $(RealPath "${BASH_SOURCE[0]}"))

echo $SIM_PATH

simInd=1

while [ $simInd -lt 101 ]
do 

  echo "Now running simulation "$simInd

  # -----------------------------------------------------------------------
  # Submit data generation job
  # -----------------------------------------------------------------------
  fsl_sub -l $SIM_PATH/sim$simInd/simlog/ -N dataGen$simInd \
          bash $SIM_PATH/generateData.sh $SIM_PATH $simInd > /tmp/$$ && \
          dataGenID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)

  # This loop waits for the data generation job to finish before
  # deciding how many batches to run. It also checks to see if the data
  # generation job has errored.
  nb=0
  i=0
  while [ $nb -lt 1 ]
  do

    # obtain number of batches
    sleep 1
    if [ -s $SIM_PATH/sim$simInd/data/nb.txt ]; then
      typeset -i nb=$(cat $SIM_PATH/sim$simInd/data/nb.txt)
    fi
    i=$(($i + 1))

    # Check for error
    if [ $i -gt 30 ]; then
      errorlog="/sim$simInd/simlog/dataGen.e$dataGenID"
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
  # Run BLMM
  # -----------------------------------------------------------------------
  bash ./blmm_cluster.sh $SIM_PATH/sim$simInd/inputs.yml

  # -----------------------------------------------------------------------
  # Submit R parameter estimation job
  # -----------------------------------------------------------------------

  # Set batch index
  batchInd=0

  while [ $batchInd -lt $nb ]
  do
    # Submit nb batches and get the ids for them
    fsl_sub -j $dataGenID -l $SIM_PATH/sim$simInd/simlog/ \
            -N lmerParamEst$simInd'_'$batchInd \
            bash $SIM_PATH/lmer_paramEst.sh $simInd $batchInd $SIM_PATH > /tmp/$$ && \
            lmerParamID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$),$lmerParamID

    i=$(($i + 1))

    if [ "$lmerParamID" == "" ] ; then
      echo "Lmer parameter estimation job submission failed!"
    fi

    # Incrememnt batch index
    batchInd=$(($batchInd + 1))

  done
  
  echo "Lmer parameter estimation jobs submitted!"

  # -----------------------------------------------------------------------
  # Check if BLMM has reached the concatenation stage (so we can delete
  # simulation data)
  # -----------------------------------------------------------------------

  # -----------------------------------------------------------------------
  # Check if BLMM has finished executing
  # -----------------------------------------------------------------------

  # Variable to check if blmm finished running (see if cleanup job output
  # the analysis complete message)
  blmmran=$(cat $SIM_PATH/sim$simInd/simlog/cleanup.o* 2> /dev/null)

  # Wait for blmm to finish running
  i=0
  while [ "$blmmran"  == "" ]
  do

    # Wait a bit before checking file again
    sleep 1

    # Check to see if blmm ran
    blmmran=$(cat $SIM_PATH/sim$simInd/simlog/cleanup.o* 2> /dev/null)

    # Update i
    i=$(($i + 1))

    # Timeout
    if [ $i -gt 10000 ]; then
      echo "Something seems to be taking a while. Please check for errors."
    fi

  done

  # -----------------------------------------------------------------------
  # Cleanup simulation
  # -----------------------------------------------------------------------

  fsl_sub -j $lmerParamID -l $SIM_PATH/sim$simInd/simlog/ -N cleanup$simInd \
          bash $SIM_PATH/cleanup.sh $SIM_PATH $simInd > /tmp/$$ && \
          cleanupID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
    
  # -----------------------------------------------------------------------
  # Remove log directory
  # -----------------------------------------------------------------------

  # Variable to check if simulation cleanup finished running 
  simcleanran=$(cat $SIM_PATH/sim$simInd/simlog/cleanup$simInd.o* 2> /dev/null)

  # Wait for blmm to finish running
  i=0
  while [ "$simcleanran"  == "" ]
  do

    # Wait a bit before checking file again
    sleep 1

    # Check to see if blmm ran
    simcleanran=$(cat $SIM_PATH/sim$simInd/simlog/cleanup$simInd.o* 2> /dev/null)

    # Update i
    i=$(($i + 1))

    # Timeout
    if [ $i -gt 10000 ]; then
      echo "Something seems to be taking a while. Please check for errors."
    fi

  done

  # Remove simulation log and inputs file (there will now be a copy of this
  # in the BLMM folder anyway)
  rm -rf $SIM_PATH/sim$simInd/simlog/
  rm -rf $SIM_PATH/sim$simInd/inputs.yml

  echo "Simulation "$simInd" ran."

  simInd=$(($simInd + 1))

done
