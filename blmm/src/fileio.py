import os
import pandas as pd
import nibabel as nib
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This file contains all miscellaneous file i/o functions used by BLMM. These
# functions exist so that basic file handling does not take too much space in
# the bulk of the main code.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Tom Maullin (Last edited 05/04/2020)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ============================================================================
#
# The below function is shorthand for reading in the following file formats:
#
#                csv, tsv, txt, dat, nii, nii.gz, img, img.gz 
#
# ============================================================================
def loadFile(filepath):

    # If the file is text data in the form of csv, tsv, txt or dat
    if filepath.lower().endswith(('.csv', '.tsv', '.txt', '.dat')):

        data = pd.io.parsers.read_csv(filepath, header=None).values

        # If we have rows and columns we should check for row and column headers:
        if data.shape[0]>1 and data.shape[1]>1:

            # Checking for column headers.
            if isinstance(data[0,0], str) and isinstance(data[0,1], str):

                # Then checking for row headers aswell
                if isinstance(data[1,0], str):
                    # Check if we have numbers in the first column,
                    # if not remove the first column because it must be 
                    # a header.
                    try:
                        float(data[1,0])
                        data = pd.io.parsers.read_csv(filepath).values
                    except:
                        data = pd.io.parsers.read_csv(
                            filepath,usecols=range(1,data.shape[1])).values
                else:
                    data = pd.io.parsers.read_csv(
                        filepath).values

            elif np.isnan(data[0,0]) and isinstance(data[0,1], str):

                # Then checking for row headers aswell
                if isinstance(data[1,0], str):
                    # Check if we have numbers in the first column,
                    # if not remove the first column because it must be 
                    # a header.
                    try:
                        float(data[1,0])
                        data = pd.io.parsers.read_csv(filepath).values
                    except:
                        data = pd.io.parsers.read_csv(
                            filepath,usecols=range(1,data.shape[1])).values
                else:
                    data = pd.io.parsers.read_csv(
                        filepath).values
                

            # Checking for row headers instead.
            elif isinstance(data[1,0], str):

                # Check if we have numbers in the first column,
                # if not remove the first column because it must be 
                # a header.
                try:
                    float(data[1,0])
                except:
                    data = pd.io.parsers.read_csv(
                        filepath,usecols=range(1,data.shape[1])).values

            # If we have a nan but numeric headers for both remove the column header by default
            elif np.isnan(data[0,0]):

                data = pd.io.parsers.read_csv(
                                    filepath).values

        # If we have more than one row but only one column, check for a column header
        elif data.shape[0]>1:
            if isinstance(data[0,0], str):
                data = pd.io.parsers.read_csv(filepath, header=None).values
            elif np.isnan(data[0,0]):
                data = pd.io.parsers.read_csv(filepath, header=None).values
        # If we have more than one column but only one row, check for a row header
        elif data.shape[1]>1:
            if isinstance(data[0,0], str):
                data = pd.io.parsers.read_csv(
                        filepath,usecols=range(1,data.shape[1])).values
            elif np.isnan(data[0,0]):
                data = pd.io.parsers.read_csv(
                        filepath,usecols=range(1,data.shape[1])).values

    # If the file is a brain image in the form of nii, nii.gz, img or img.gz
    else:
        # If the file exists load it.
        try:
            data = nib.load(filepath)
        except:
            try:
                if os.path.isfile(os.path.join(filepath, '.nii.gz')):
                    data = nib.load(os.path.join(filepath, '.nii.gz'))
                elif os.path.isfile(os.path.join(filepath, '.nii')):
                    data = nib.load(os.path.join(filepath, '.nii'))
                elif os.path.isfile(os.path.join(filepath, '.img.gz')):
                    data = nib.load(os.path.join(filepath, '.img.gz'))
                else:
                    data = nib.load(os.path.join(filepath, '.img'))
            except:
                raise ValueError('Input file not found: ' + str(filepath))

    return data


# ============================================================================
#
# The below function takes in a string representing a contrast vector and 
# returns the contrast vector as an array.
#
# ============================================================================
def str2vec(c):

    c = str(c)
    c = c.replace("'", "")
    c = c.replace('][', '], [').replace('],[', '], [').replace('] [', '], [')
    c = c.replace('[ [', '[[').replace('] ]', ']]')
    cs = c.split(' ')
    cf = ''
    for i in range(0,len(cs)):
        cs[i]=cs[i].replace(',', '')
        cf=cf + cs[i]
        if i < (len(cs)-1):
            cf = cf + ', '
        
    return(eval(cf))


# ============================================================================
#
# The below function adds a block of voxels to a pre-existing NIFTI or creates
# a NIFTI of specified dimensions if not.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `fname`: An absolute path to the Nifti file.
# - `block`: The block of values to write to the NIFTI.
# - `blockInds`: The indices representing the 3D coordinates `block` should be 
#                written to in the NIFTI. (Note: It is assumed if the NIFTI is
#                4D we assume that the indices we want to write to in each 3D
#                volume/slice are the same across all 3D volumes/slices).
# - `dim` (optional): If creating the NIFTI image for the first time, the 
#                     dimensions of the NIFTI image must be specified.
# - `volInd` (optional): If we only want to write to one 3D volume/slice,
#                        within a 4D file, this specifies the index of the
#                        volume of interest.
# - `aff` (optional): If creating the NIFTI image for the first time, the 
#                     affine of the NIFTI image must be specified.
# - `hdr` (optional): If creating the NIFTI image for the first time, the 
#                     header of the NIFTI image must be specified.
#
# ============================================================================
def addBlockToNifti(fname, block, blockInds,dim=None,volInd=None,aff=None,hdr=None):

    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            f = os.open(fname + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True

    # Check volInd is correct datatype
    if volInd is not None:

        volInd = int(volInd)

    # Check whether the NIFTI exists already
    if os.path.isfile(fname):

        # Load in NIFTI
        img = nib.load(fname)

        # Work out dim if we don't already have it
        dim = img.shape

        # Work out data
        data = img.get_fdata()

        # Work out affine
        affine = img.affine
        
    else:

        # If we know how, make the NIFTI
        if dim is not None:
            
            # Make data
            data = np.zeros(dim)

            # Make affine
            if aff is None:
                affine = np.eye(4)
            else:
                affine = aff

        else:

            # Throw an error because we don't know what to do
            raise Exception('NIFTI does not exist and dimensions not given')

    # Work out the number of output volumes inside the nifti 
    if len(dim)==3:

        # We only have one volume in this case
        n_vol = 1
        dim = np.array([dim[0],dim[1],dim[2],1])

    else:

        # The number of volumes is the last dimension
        n_vol = dim[3]

    # Seperate copy of data for outputting
    data_out = np.array(data).reshape(dim)

    # Work out the number of voxels
    n_vox = np.prod(dim[:3])

    # Reshape     
    data = data.reshape([n_vox, n_vol])

    # Add all the volumes
    if volInd is None:

        # Add block
        data[blockInds,:] = block.reshape(data[blockInds,:].shape)
        
        # Cycle through volumes, reshaping.
        for k in range(0,data.shape[1]):

            data_out[:,:,:,k] = data[:,k].reshape(int(dim[0]),
                                                  int(dim[1]),
                                                  int(dim[2]))

    # Add the one volume in the correct place
    else:

        # We're only looking at this volume
        data = data[:,volInd].reshape((n_vox,1))

        # Add block
        data[blockInds,:] = block.reshape(data[blockInds,:].shape)
        
        # Put in the volume
        data_out[:,:,:,volInd] = data[:,0].reshape(int(dim[0]),
                                                 int(dim[1]),
                                                 int(dim[2]))


    # Make NIFTI
    nifti = nib.Nifti1Image(data_out, affine, header=hdr)
    
    # Save NIFTI
    nib.save(nifti, fname)

    # Delete lock file, so other jobs know they can now write to the
    # file
    os.remove(fname + ".lock")
    os.close(f)

    del nifti, fname, data_out, affine


# ============================================================================
#
# The below function reads in a numpy file as a memory map and returns the 
# specified lines from the file. The benefit of this is that the entire file
# is not read into memory and we retrieve only the parts of the file we need.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `filename`: The name of the file.
# - `lines`: Indices of the lines we wish to retrieve from the file.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `data_lines`: The lines from the file we wanted to retrieve.  
#
# ============================================================================
def readLinesFromNPY(filename, lines):

    # Load in the file but in memory map mode
    data_memmap = np.load(filename,mmap_mode='r')

    # Read in the desired lines
    data_lines = np.array(data_memmap[lines,:])

    # We don't want this file hanging around
    del data_memmap

    return(data_lines)


# ============================================================================
#
# The below function takes in an analysis mask and either; returns the mask in
# flattened form, or, given a specified block number, vb, and number of blocks,
# returns a mask, in flattened form, containing only the indices of the vb^th
# block of voxels.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `am`: The analysis mask as a 3d volume.
# - `vb`: The number of the voxel block of interest (less than 0 return all
#         indices)
# - `nvb`: The number of voxel blocks in total.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `amInds`: The indices for the mask.
#
# ============================================================================
def get_amInds(am, vb=None, nvb=None):

  # Reshape the analysis mask
  am = am.reshape([np.prod(am.shape),1])

  # Work out analysis mask indices.
  amInds=np.where(am==1)[0]

  # Get vb^th block of voxels
  if vb is not None and vb>=0:

    # Split am into equal nvb "equally" (ish) sized blocks and take
    # the vb^th block.
    amInds = np.array_split(amInds, nvb)[vb]

  return(amInds)


# ============================================================================
#
# The below function computes the  number of voxel blocks we have to split the
# data into, due to memory constraints.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `inputs`: The inputs dictionary read from a blmm inputs cfg file.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `nvb`: The number of voxel blocks we have to split the data into in order
#          to compute the design.
#
# ============================================================================
def numVoxelBlocks(inputs):

  # ----------------------------------------------------------------
  # Number of levels and number of random effects
  # ----------------------------------------------------------------
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

  # Check if the maximum memory is saved.    
  if 'MAXMEM' in inputs:
    MAXMEM = eval(inputs['MAXMEM'])
  else:
    MAXMEM = 2**32

  # ----------------------------------------------------------------
  # Work out number of voxels we'd ideally want in a block
  # ----------------------------------------------------------------
  # This is done by taking the maximum memory (in bytes), divided
  # by roughly the amount of memory a float in numpy takes (8), 
  # divided by 10 (allowing us to have up to 10 variables of
  # allowed size at any one time), divided by q^2 (the number of 
  # random effects squared/the largest matrix size we would
  # look at).
  vPerBlock = MAXMEM/(10*8*(q**2))

  # Read in analysis mask (if present)
  if 'analysis_mask' in inputs:
    am = loadFile(inputs['analysis_mask'])
    am = am.get_fdata()
  else:

    # --------------------------------------------------------------
    # Get one Y volume to make full Nifti mask
    # --------------------------------------------------------------

    # Y volumes
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

    # Get mask of ones
    am = np.ones(Y0.shape)

  # Work out number of non-zero voxels in analysis mask
  v = np.sum(am!=0)

  # Work out number of voxel blocks we would need.
  nvb = v//vPerBlock+1

  # Return number of voxel blocks
  return(nvb)


# ============================================================================
#
# The below function computes the  number of voxel blocks we are able to split
# the data into for parallel computation.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `inputs`: The inputs dictionary read from a blmm inputs cfg file.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `nvb`: The number of voxel blocks we can split the data into for parallel
#          computation.
#
# ============================================================================
def pracNumVoxelBlocks(inputs):

  # Check if maximum number of voxel blocks specified,
  # otherwise, default to 60
  if 'maxnvb' in inputs:
    maxnvb = inputs['maxnvb']
  else:
    maxnvb = 60

  # Work out number of voxel blocks we should use.
  nvb = np.min([numVoxelBlocks(inputs), maxnvb])

  # Return number of voxel blocks
  return(nvb)
