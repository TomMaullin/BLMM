#!/bin/bash                                                                                                                                                                                    
#$ -o $HOME/logpFS                                                                                                                                                                               
#$ -e $HOME/logpFS

source $FSLDIR/fslpython/bin/activate ./blmmenv
python -c "from lib import simRF_DAC; simRF_DAC.main()"
