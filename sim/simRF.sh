#!/bin/bash                                                                                                                                                                                    

source $FSLDIR/fslpython/bin/activate ./blmmenv
python -c "from lib import simRF_BC; simRF_BC.main()"
