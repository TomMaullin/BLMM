OutDir=$1

# -----------------------------------------------------------------------
# Work out LMM path
# -----------------------------------------------------------------------
RealPath() {
    (echo $(cd $(dirname "$1") && pwd -P)/$(basename "$1"))
}

LMM_PATH=$(dirname $(RealPath "${BASH_SOURCE[0]}"))

fsl_sub -l log/ -N sim1 bash $LMM_PATH/sim/LMMPaperSim.sh $OutDir 1


qsub $LMM_PATH/sim/simlmer.R 