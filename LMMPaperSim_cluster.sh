OutDir=$1

# -----------------------------------------------------------------------
# Work out LMM path
# -----------------------------------------------------------------------
RealPath() {
    (echo $(cd $(dirname "$1") && pwd -P)/$(basename "$1"))
}

LMM_PATH=$(dirname $(RealPath "${BASH_SOURCE[0]}"))

#fsl_sub -l log/ -N sim1 bash $LMM_PATH/sim/LMMPaperSim.sh $OutDir 1


#qsub $LMM_PATH/sim/LMMPaperSim.R 


# The first job must save it's setup ID
fsl_sub -l log/ -N sim1 bash $LMM_PATH/sim/LMMPaperSim.sh 1 > /tmp/$$ && setupID=$(awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}' /tmp/$$)

# Run all other simulations as seperate jobs only on hold based on the first job
sims=1000
i=2
while [ $i -le $sims ]
do

	fsl_sub -j $setupID -l log/ -N sim$i bash $LMM_PATH/sim/LMMPaperSim.sh $i
	i=$(($i + 1))

done
