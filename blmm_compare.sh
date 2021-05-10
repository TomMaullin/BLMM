#!/bin/bash

echo "Submitting model comparison..."

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
# Read and parse BLMM config files
# -----------------------------------------------------------------------
# Read first config
cfg1=$(RealPath $1/inputs.yml)

# Read second config
cfg1=$(RealPath $2/inputs.yml)

# -----------------------------------------------------------------------
# Make output directory
# -----------------------------------------------------------------------
# Read output directory
outdir=$(RealPath $3)

# Make directory
mkdir -p $outdir

# -----------------------------------------------------------------------
# Make a copy of the input files, if the user touches the cfg file this
# will not mess with anything now.
# -----------------------------------------------------------------------
inputs1=$outdir/inputs_model1.yml
cp $cfg1 $inputs1

inputs2=$outdir/inputs_model2.yml
cp $cfg2 $inputs2

# -----------------------------------------------------------------------
# Submit model comparison job
# -----------------------------------------------------------------------
fsl_sub -l $outdir/log -N compare bash $BLMM_PATH/scripts/cluster_blmm_compare.sh $inputs1 $inputs2 $outdir > /tmp/$$ && compareID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)
if [ "$compareID" == "" ] ; then
  echo "Model comparison job submission failed!"
else
  echo "Submitted: Model comparison job."
fi
