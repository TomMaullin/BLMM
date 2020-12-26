import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import scipy
import scipy.sparse
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
from scipy import ndimage
import time
import pandas as pd


# ===========================================================================
#
# Inputs:
#
# ---------------------------------------------------------------------------
#
# ===========================================================================
def cleanupR(dim,OutDir,simNo):

	# -----------------------------------------------------------------------
    # Get simulation directory
	# -----------------------------------------------------------------------
	# Simulation directory
    simDir = os.path.join(OutDir, 'sim' + str(simNo))

	# -----------------------------------------------------------------------
    # Read in design in BLMM inputs form (this just is easier as code already
    # exists for using this format).
	# -----------------------------------------------------------------------
    # There should be an inputs file in each simulation directory
    with open(os.path.join(simDir,'inputs.yml'), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

	# -----------------------------------------------------------------------
    # Get number of random effects, levels and random factors in design
	# -----------------------------------------------------------------------
	# Random factor variables.
	rfxmats = inputs['Z']

	# Number of random effects
	r = len(rfxmats)

	# Number of random effects for each factor, q
	nraneffs = []

	# Number of levels for each factor, l
	nlevels = []

	for k in range(r):

	    rfxdes = loadFile(rfxmats[k]['f' + str(k+1)]['design'])
	    rfxfac = loadFile(rfxmats[k]['f' + str(k+1)]['factor'])

	    nraneffs = nraneffs + [rfxdes.shape[1]]
	    nlevels = nlevels + [len(np.unique(rfxfac))]

	# Get number of random effects
	nraneffs = np.array(nraneffs)
	nlevels = np.array(nlevels)
	q = np.sum(nraneffs*nlevels)


	# Get number of observations and fixed effects
	X = pd.io.parsers.read_csv(os.path.join(simDir,"data","X.csv"), header=None).values
	n = X.shape[0]
	p = X.shape[1]