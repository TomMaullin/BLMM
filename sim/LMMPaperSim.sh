#!/bin/bash
fslpython -c "from sim import LMMPaperSim; sim2D(OutDir='$1',desInd=int('$2'))"