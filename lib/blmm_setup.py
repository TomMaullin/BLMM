import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import warnings
import resource
import nibabel as nib
import sys
import os
import shutil
import yaml
from lib.blm_eval import blm_eval
from lib.blm_load import blm_load

# Main takes in two arguments at most:
# - input: Either the path to an input file or an input structure
#          with all paths already set to absolute.
# - retnb: A boolean which tells us whether to return the number
#          of batches needed (retnb=True) or save the variable
#          in a text file (retnb=False).
def main(*args):

    # Change to blm directory
    pwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if len(args)==0 or (not args[0]):
        # Load in inputs
        ipath = os.path.abspath(os.path.join('..','blm_config.yml'))
        with open(ipath, 'r') as stream:
            inputs = yaml.load(stream)
        retnb = False
    else:
        if type(args[0]) is str:
            if os.path.isabs(args[0]):
                ipath = args[0]
            else:
                ipath = os.path.abspath(os.path.join(pwd, args[0]))
            # In this case inputs file is first argument
            with open(ipath, 'r') as stream:
                inputs = yaml.load(stream)
                # Work out whether to return nb or save it in a 
                # file
                if len(args)>1:
                    retnb = args[1]
                else:
                    retnb = False
        else:  
            # In this case inputs structure is first argument.
            inputs = args[0]
            ipath = ''
            retnb = True

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

        if not os.path.isabs(inputs['outdir']):

            # Change output directory in inputs
            inputs['outdir'] = os.path.join(pwd, inputs['outdir'])

        # Update inputs
        with open(ipath, 'w') as outfile:
            yaml.dump(inputs, outfile, default_flow_style=False)

    # Change paths to absoluate if they aren't already    
    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    OutDir = inputs['outdir']

    # Get number of parameters
    c1 = blm_eval(inputs['contrasts'][0]['c' + str(1)]['vector'])
    c1 = np.array(c1)
    n_p = c1.shape[0]
    del c1

    # Make output directory and tmp
    if not os.path.isdir(OutDir):
        os.mkdir(OutDir)
    if not os.path.isdir(os.path.join(OutDir, "tmp")):
        os.mkdir(os.path.join(OutDir, "tmp"))

    with open(inputs['Y_files']) as a:

        Y_files = []
        i = 0
        for line in a.readlines():

            Y_files.append(line.replace('\n', ''))

    # Load in one nifti to check NIFTI size
    try:
        Y0 = blm_load(Y_files[0])
    except Exception as error:
        raise ValueError('The NIFTI "' + Y_files[0] + '"does not exist')

    # Get the maximum memory a NIFTI could take in storage. We divide by 3
    # as approximately a third of the volume is actually non-zero/brain
    NIFTIsize = sys.getsizeof(np.zeros(Y0.shape,dtype='uint64'))

    if NIFTIsize > MAXMEM:
        raise ValueError('The NIFTI "' + Y_files[0] + '"is too large')

    # Similar to blksize in SwE, we divide by 8 times the size of a nifti
    # to work out how many blocks we use. We divide NIFTIsize by 3
    # as approximately a third of the volume is actually non-zero/brain 
    # and then also divide though everything by the number of parameters
    # in the analysis.
    blksize = np.floor(MAXMEM/8/NIFTIsize/n_p)
    if blksize == 0:
        raise ValueError('Blocksize too small.')

    # Check F contrast ranks 
    n_c = len(inputs['contrasts'])
    for i in range(0,n_c):

        # Read in contrast vector
        cvec = blm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        cvec = np.array(cvec)

        if cvec.ndim>1:
                
            # Get dimension of cvector
            q = cvec.shape[0]

            if np.linalg.matrix_rank(cvec)<q:
                raise ValueError('F contrast: \n' + str(cvec) + '\n is not of correct rank.')

    if not retnb:
        with open(os.path.join(OutDir, "nb.txt"), 'w') as f:
            print(int(np.ceil(len(Y_files)/int(blksize))), file=f)
    else:
        return(int(np.ceil(len(Y_files)/int(blksize))))

    w.resetwarnings()

if __name__ == "__main__":
    main()
