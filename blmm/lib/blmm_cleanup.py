import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import sys
import os
import glob
import shutil
import yaml
from blmm.src.fileio import convert_all_dat_files, loadFile
np.set_printoptions(threshold=sys.maxsize)

# ====================================================================================
#
# This file is the cleanup stage of the BLMM pipeline. It simply deletes any remaining
# files that are no longer needed.
#
# ------------------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited: 04/04/2020)
#
# ------------------------------------------------------------------------------------
#
# The code takes the following inputs:
#
#  - `ipath`: Path to an `inputs` yml file, following the same formatting guidelines
#             as `blmm_config.yml`. 
#
# ====================================================================================
def cleanup(ipath):

    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Inputs file is first argument
    with open(os.path.join(ipath), 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # --------------------------------------------------------------------------------
    # Read Output directory, work out number of batches
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']
    
    # Save a copy of inputs in output directory
    with open(os.path.join(OutDir, 'inputs.yml'), 'w') as outfile:
        yaml.dump(inputs, outfile, default_flow_style=False)

    # --------------------------------------------------------------------------------
    # Clean up files
    # --------------------------------------------------------------------------------
    os.remove(os.path.join(OutDir, 'nb.txt'))
    if os.path.exists(os.path.join(OutDir, 'nvb.txt')):
        os.remove(os.path.join(OutDir, 'nvb.txt'))
    shutil.rmtree(os.path.join(OutDir, 'tmp'))
    
    # --------------------------------------------------------------------------------
    # Convert all output files to the appropriate filetype
    # --------------------------------------------------------------------------------
    # Read in an input image for reference
    # Read in the Y_files (make sure to remove new line characters)
    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Convert all files to same file type
    convert_all_dat_files(OutDir, Y_files[0])

    # Check if we are protecting disk quota.
    if 'diskMem' in inputs:

        if inputs['diskMem']==1:

            # If we are then this means we have only run one part of the image
            # during this run. Check which part it was and delete that so it
            # isn't run next time.
            memmaskFiles = glob.glob(os.path.join(OutDir, 'blmm_vox_memmask*.dat'))
            os.remove(memmaskFiles[0])



    print('Analysis complete!')
    print('')
    print('---------------------------------------------------------------------------')
    print('')
    print('Check results in: ', OutDir)