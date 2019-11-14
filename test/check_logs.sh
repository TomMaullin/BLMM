# Work out BLM dir
BLMdir=$(realpath ../)
cd $BLMdir

# Run each test case
i=1
for cfgIDs in $(ls ./test/cfgids/*)
do
  echo " "
  echo '================================================================'
  testcase=$(basename $cfgIDs)
  testcase=$(echo $testcase | sed "s/IDs//g")
  echo "Error logs for testcase $testcase"
  echo " "
  cfgIDs=$(realpath $cfgIDs)

  # Read IDs from each line of file
  while read LINE; do 
    IDs=$IDs" "$LINE; 
  done < $cfgIDs

  # Status update
  for ID in $IDs
  do
    logfile=$(ls ./log/*.e$ID | head -1)
    logfilecontent=$(cat $logfile)
    if [ ! -z "$logfilecontent" ] ; then
      echo "Logged error in: $logfile"
      echo $logfilecontent
      logsencountered=1
      echo " "
    fi
  done

  if [ -z "$logsencountered" ] ; then
    echo "No errors encountered for testcase $testcase"
  fi

  IDs=''
  i=$(($i + 1))
  logsencountered=""
done
