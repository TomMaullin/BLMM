# Work out BLM dir
BLMdir=$(realpath ../)
cd $BLMdir

# Include the parse yaml function
. scripts/parse_yaml.sh
cd $BLMdir/test

# Read the test directory
gtdir=$1
if [ -v $gtdir ] ; then
  echo "Please enter ground truth directory."
  exit
fi

# Run each test case
i=1
for cfg in $(ls ./cfg/test_cfg*_copy.yml)
do
  
  cfgfilepath=$(realpath $cfg)
  cfgname=$(basename $(echo $cfg | sed "s/\_copy//g"))

  echo " "
  echo " "
  echo "Now verifying: $cfgname".
  echo " "
  echo " "

  # read yaml file to get output directory
  eval $(parse_yaml $cfgfilepath "config_")
  
  fslpython -c "import verify_test_cases; verify_test_cases.main('$config_outdir', '$gtdir/$(basename $config_outdir)')"

  i=$(($i + 1))
done
