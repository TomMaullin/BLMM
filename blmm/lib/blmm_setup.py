import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import sys
import os
import glob
import shutil
import yaml
from blmm.src.fileio import loadFile, str2vec, pracNumVoxelBlocks, get_amInds, addBlockToNifti

# ====================================================================================
#
# This file is the first stage of the BLMM pipeline. It reads in the inputs file and
# reformats it where necessary, then works out how many batches should be used for 
# computation. The number of batches is output into the file `nb.txt` which informs 
# the rest of the blmm pipeline how many batches are required for computation. 
#
# ------------------------------------------------------------------------------------
#
# Author: Tom Maullin (Last edited: 18/06/2020)
#
# ------------------------------------------------------------------------------------
#
# The code takes the following inputs:
#
#  - input path (optional): If specified, the first argument will be assumed to be a
#                           path to an `inputs` yml file, following the same 
#                           formatting guidelines as `blmm_config.yml`. If not 
#                           specified, the default file `blmm_config.yml` will be 
#                           assumed to contain the inputs.
#
# ------------------------------------------------------------------------------------
#
# Developer Note: As of 18/06/2020, this code contains a backdoor option `diskMem` 
#                 which can be added into the `inputs.yml` file and has not been made
#                 available to users. The purpose of this option is that for large 
#                 designs the `ZtY` file produced may be too big to save as a file. 
#                 To overcome this, the diskMem (or 'disk memory') option splits the
#                 brain mask up voxelwise into `blmm_memmask` files. The analysis can
#                 now be run one "chunk" of the brain at a time by repeat calls to 
#                 `blmm_cluster.sh` in serial. Whilst this hasn't been made available
#                 to users (as we are yet to see a design that really needs it), it is
#                 very useful when developing code to be able to run small chunks of 
#                 the brain instead of the whole thing. For this reason it has been 
#                 left in here. However, it is not maintained, so may need some
#                 tweaking to use.
#
# ====================================================================================
def setup(*args):

    # Change to blmm directory
    pwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Check the inputs
    if len(args)==0 or (not args[0]):
        # Load in inputs
        ipath = os.path.abspath(os.path.join('..','blmm_config.yml'))
        with open(ipath, 'r') as stream:
            inputs = yaml.load(stream,Loader=yaml.FullLoader)
    else:
        if os.path.isabs(args[0]):
            ipath = args[0]
        else:
            ipath = os.path.abspath(os.path.join(pwd, args[0]))
        # In this case inputs file is first argument
        with open(ipath, 'r') as stream:
            inputs = yaml.load(stream,Loader=yaml.FullLoader)

    # Save absolute filepaths in place of relative filepaths
    if ipath: 

        # Y files
        if not os.path.isabs(inputs['Y_files']):

            # Change Y in inputs
            inputs['Y_files'] = os.path.join(pwd, inputs['Y_files'])

        # If mask files are specified
        if 'data_mask_files' in inputs:

            # M_files
            if not os.path.isabs(inputs['data_mask_files']):

                # Change M in inputs
                inputs['data_mask_files'] = os.path.join(pwd, inputs['data_mask_files'])

        # If analysis mask file specified,        
        if 'analysis_mask' in inputs:

            # M_files
            if not os.path.isabs(inputs['analysis_mask']):

                # Change M in inputs
                inputs['analysis_mask'] = os.path.join(pwd, inputs['analysis_mask'])

        # If X is specified
        if not os.path.isabs(inputs['X']):

            # Change X in inputs
            inputs['X'] = os.path.join(pwd, inputs['X'])

        # If the output directory is not an absolute, assume its the relative
        # to the present working directory 
        if not os.path.isabs(inputs['outdir']):

            # Change output directory in inputs
            inputs['outdir'] = os.path.join(pwd, inputs['outdir'])

        # Change each random effects factor
        nf = len(inputs['Z'])
        for i in range(0,nf):

            inputs['Z'][i]['f' + str(i+1)]['factor'] = os.path.join(pwd, inputs['Z'][i]['f' + str(i+1)]['factor'])
            inputs['Z'][i]['f' + str(i+1)]['design'] = os.path.join(pwd, inputs['Z'][i]['f' + str(i+1)]['design'])

        # Update inputs
        with open(ipath, 'w') as outfile:
            yaml.dump(inputs, outfile, default_flow_style=False)

    # Check if the maximum memory is saved.    
    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    # Output directory
    OutDir = inputs['outdir']

    # --------------------------------------------------------------------------------
    # Remove any files from the previous runs
    #
    # Note: This is important as if we are outputting blocks to files we want to be
    # sure none of the previous results are lingering around anywhere.
    # --------------------------------------------------------------------------------

    # We don't do it if we are in diskMem mode though, as in this mode we run several
    # smaller analyses instead of one large one in order to preserve disk memory.
    if 'diskMem' not in inputs:

        files = ['blmm_vox_n.nii', 'blmm_vox_mask.nii', 'blmm_vox_edf.nii', 'blmm_vox_beta.nii',
                 'blmm_vox_llh.nii', 'blmm_vox_sigma2.nii', 'blmm_vox_D.nii', 'blmm_vox_resms.nii',
                 'blmm_vox_cov.nii', 'blmm_vox_conT_swedf.nii', 'blmm_vox_conT.nii', 'blmm_vox_conTlp.nii',
                 'blmm_vox_conSE.nii', 'blmm_vox_con.nii', 'blmm_vox_conF.nii', 'blmm_vox_conF_swedf.nii',
                 'blmm_vox_conFlp.nii', 'blmm_vox_conR2.nii']

    else:

        # Check if this is the first run of the disk memory code
        if inputs['diskMem']==1 and not glob.glob(os.path.join(OutDir, 'blmm_vox_memmask*.nii')):

            files = ['blmm_vox_n.nii', 'blmm_vox_mask.nii', 'blmm_vox_edf.nii', 'blmm_vox_beta.nii',
                     'blmm_vox_llh.nii', 'blmm_vox_sigma2.nii', 'blmm_vox_D.nii', 'blmm_vox_resms.nii',
                     'blmm_vox_cov.nii', 'blmm_vox_conT_swedf.nii', 'blmm_vox_conT.nii', 'blmm_vox_conTlp.nii',
                     'blmm_vox_conSE.nii', 'blmm_vox_con.nii', 'blmm_vox_conF.nii', 'blmm_vox_conF_swedf.nii',
                     'blmm_vox_conFlp.nii', 'blmm_vox_conR2.nii']

        else:

            files = []


    for file in files:
        if os.path.exists(os.path.join(OutDir, file)):
            os.remove(os.path.join(OutDir, file))

    if os.path.exists(os.path.join(OutDir, 'tmp')):  
        shutil.rmtree(os.path.join(OutDir, 'tmp'))

    # Get number of parameters
    L1 = str2vec(inputs['contrasts'][0]['c' + str(1)]['vector'])
    L1 = np.array(L1)
    p = L1.shape[0]
    del L1

    # Make output directory and tmp
    if not os.path.isdir(OutDir):
        os.mkdir(OutDir)
    if not os.path.isdir(os.path.join(OutDir, "tmp")):
        os.mkdir(os.path.join(OutDir, "tmp"))

    # Read in the Y_files (make sure to remove new line characters)
    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load in one nifti to check NIFTI size
    try:
        Y0 = loadFile(Y_files[0])
    except Exception as error:
        raise ValueError('The NIFTI "' + Y_files[0] + '"does not exist')

    # Get an estimate of the maximum memory a NIFTI could take in storage.
    NIFTImem = sys.getsizeof(np.zeros(Y0.shape,dtype='uint64'))

    if NIFTImem > MAXMEM:
        raise ValueError('The NIFTI "' + Y_files[0] + '"is too large')

    # --------------------------------------------------------------------------------
    # Get q and v
    # --------------------------------------------------------------------------------
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

    # Save q (useful to have around)
    inputs["q"] = str(q)

    # --------------------------------------------------------------------------------
    # Safe mode
    # --------------------------------------------------------------------------------
    # Check if we are in safe mode.
    if 'safeMode' not in inputs:
        inputs['safeMode']=1
        
    # Update inputs
    with open(ipath, 'w') as outfile:
        yaml.dump(inputs, outfile, default_flow_style=False)

    # Get v
    NIFTIsize = Y0.shape
    v = int(np.prod(NIFTIsize))

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use. We also divide though everything
    # by the number of parameters in the analysis.
    blksize = int(np.floor(MAXMEM/8/NIFTImem/p))
  
    if blksize == 0:
        raise ValueError('Blocksize too small.')

    # Check F contrast ranks 
    n_c = len(inputs['contrasts'])
    for i in range(0,n_c):

        # Read in contrast vector
        cvec = str2vec(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        cvec = np.array(cvec)

        if cvec.ndim>1:

            # Check the F contrast has appropriate rank
            if np.linalg.matrix_rank(cvec)<cvec.shape[0]:
                raise ValueError('F contrast: \n' + str(cvec) + '\n is not of correct rank.')

    # Output number of batches to a text file
    with open(os.path.join(OutDir, "nb.txt"), 'w') as f:
        print(int(np.ceil(len(Y_files)/int(blksize))), file=f)

    # Check if we are protecting disk quota as well.
    if 'diskMem' in inputs:

        if inputs['diskMem']==1:

            # Check if this is the first run of the disk memory code
            if not glob.glob(os.path.join(OutDir, 'blmm_vox_memmask*.nii')):

                # --------------------------------------------------------------------------------
                # Read Mask 
                # --------------------------------------------------------------------------------
                if 'analysis_mask' in inputs:

                    amask_path = inputs["analysis_mask"]
                    
                    # Read in the mask nifti.
                    amask = loadFile(amask_path).get_fdata().reshape([v,1])

                else:

                    # By default make amask ones
                    amask = np.ones([v,1])

                # Get indices for whole analysis mask. 
                amInds = get_amInds(amask)

                # ------------------------------------------------------------------------
                # Split the voxels into computable groups
                # ------------------------------------------------------------------------

                # Work out the number of voxels we can actually save at a time (rough
                # guess).
                nvs = MAXMEM/(150*q)

                # Work out number of groups we have to split indices into.
                nvg = int(len(amInds)//nvs+1)

                # Split voxels we want to look at into groups we can compute
                voxelGroups = np.array_split(amInds, nvg)

                # Loop through list of voxel indices, saving each group of voxels, in
                # turn.
                for cv in range(nvg):

                    # Save the masks for each block
                    addBlockToNifti(os.path.join(OutDir, 'blmm_vox_memmask'+str(cv+1)+'.nii'), np.ones(len(voxelGroups[cv])), voxelGroups[cv],volInd=0,dim=NIFTIsize,aff=Y0.affine,hdr=Y0.header)

            # --------------------------------------------------------------------------------
            # Set the analysis mask to the first one that comes up with ls, run that and then
            # delete it during cleanup ready for the next run.
            # --------------------------------------------------------------------------------

            # Get the analysis masks
            memmaskFiles = glob.glob(os.path.join(OutDir, 'blmm_vox_memmask*.nii'))

            # Set the analysis mask for this analysis 
            inputs['analysis_mask'] = memmaskFiles[0]
            with open(ipath, 'w') as outfile:
                yaml.dump(inputs, outfile, default_flow_style=False)

    # If in voxel batching mode, save the number of voxel batches we need
    if 'voxelBatching' in inputs:

        if inputs['voxelBatching']:

            # Obtain number of voxel blocks for parallelization.
            nvb = pracNumVoxelBlocks(inputs)

            # Output number of voxel batches to a text file
            with open(os.path.join(OutDir, "nvb.txt"), 'w') as f:
                print(int(nvb), file=f)

    # --------------------------------------------------------------------------------
    # lmer directories.
    # --------------------------------------------------------------------------------

    # If directory doesn't exist, make it
    if not os.path.exists(os.path.join(OutDir,"data")):
        os.mkdir(os.path.join(OutDir,"data"))

    # Reset warnings
    w.resetwarnings()
    
    # Return nb
    return(int(np.ceil(len(Y_files)/int(blksize))))

if __name__ == "__main__":
    main()
