#!/bin/bash
fslpython -c "from sim import LMMPaperSim; LMMPaperSim.sim2D(desInd=1, OutDir='/well/nichols/users/inf852/PaperSims')"
#fslpython -c "from sim import LMMPaperSim; LMMPaperSim.TstatisticPPplots(desInd=1, OutDir='/well/nichols/users/inf852/PaperSims')"