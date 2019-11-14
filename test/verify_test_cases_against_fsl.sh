# Work out BLM dir
BLMdir=$(realpath ../)
cd $BLMdir

# Include the parse yaml function
. scripts/parse_yaml.sh
cd $BLMdir/test

# Read the test directory
gtdir=$1

# Run each test case
i=1
for cfg in $(ls ./cfg/fsltest_cfg*_copy.yml)
do
  cfgname=$(basename $(echo $cfg | sed "s/\_copy//g"))  
  cfgfilepath=$(realpath $cfg)

  echo " "
  echo " "
  echo "Now verifying: $cfgname".
  echo " "
  echo " "

  # read yaml file to get output directory
  eval $(parse_yaml $cfgfilepath "config_")
  fslpython -c "import verify_test_cases_against_fsl; verify_test_cases_against_fsl.main('$(dirname $config_outdir)/fsl/', '$(dirname $config_outdir)/blm/')"

  i=$(($i + 1))
done
