# Work out BLM dir
BLMdir=$(realpath ../)
cd $BLMdir

# Read the test and data directories
testdir=$1
datadir=$2
if [ -v $datadir ] ; then
  echo "Please specify data directory."
  exit
fi
if [ -v $testdir ] ; then
  echo "Please specify data directory."
  exit
fi

i=1
for cfg in $(ls ./test/cfg/fsltest_cfg??.yml)
do
  cp $cfg ./test/cfg/fsltest_cfg$(printf "%.2d" $i)_copy.yml
  i=$(($i + 1))
done


# Change the name of the test and data directories in the test configurations
find ./test/cfg/fsltest_cfg*_copy.yml -type f -exec sed -i "s|TEST_DIRECTORY|$testdir|g" {} \;
find ./test/cfg/fsltest_cfg*_copy.yml -type f -exec sed -i "s|DATA_DIRECTORY|$datadir|g" {} \;


# Make a directory to store job ids if there isn't one already.
mkdir -p ./test/cfgids

# Run each test case
i=1
for cfg in $(ls ./test/cfg/fsltest_cfg*_copy.yml)
do
  cfgname=$(basename $(echo $cfg | sed "s/\_copy//g"))
  echo "Now running testcase $cfgname"
  cfgfile=$(realpath $cfg)

  # Run blm for test configuration and save the ids
  bash ./blm_cluster.sh $cfgfile IDs > ./test/cfgids/fsltestIDs$(printf "%.2d" $i)tmp

  # Remove any commas from testIDs
  sed 's/,/ /g' ./test/cfgids/fsltestIDs$(printf "%.2d" $i)tmp > ./test/cfgids/fsltestIDs$(printf "%.2d" $i)
  rm ./test/cfgids/fsltestIDs$(printf "%.2d" $i)tmp

  # Status update
  qstat
  i=$(($i + 1))
done

# Include the parse yaml function
. scripts/parse_yaml.sh

# Now run equivalent fsl analyses
i=1
for cfg in $(ls ./test/cfg/fsltest_cfg*_copy.yml)
do
  # Obtain output directory
  cfgfile=$(realpath $cfg)
  eval $(parse_yaml $cfgfile "config_")

  echo "Running fsl_glm for analysis $i"

  mkdir -p $(dirname $config_outdir)
  mkdir -p $(dirname $config_outdir)/fsl/

  fsl_glm -i $datadir/${config_ns}subNifti.nii.gz -d $datadir/X_${config_ns}_fslformat.txt -o $(dirname $config_outdir)/fsl/fsl_vox_betas -c $datadir/C_fslformat.txt --out_t=$(dirname $config_outdir)/fsl/fsl_vox_Tstat_c --out_f=$(dirname $config_outdir)/fsl/fsl_vox_Fstat_c --out_varcb=$(dirname $config_outdir)/fsl/fsl_vox_cov_c
  i=$(($i + 1))
done
