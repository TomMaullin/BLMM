#!/bin/bash
$FSLDIR/fslpython/bin/conda create python=3.6 -p ./myenv
source $FSLDIR/fslpython/bin/activate ./myenv
pip install nilearn
fslpython -c "from lib import simRF; simRF.main()"
