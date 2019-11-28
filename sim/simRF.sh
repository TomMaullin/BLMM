#!/bin/bash                                                                                                                                                                                    

source $FSLDIR/fslpython/bin/activate ./blmmenv
python -c "from lib import simRF_L; simRF_L.main()"
