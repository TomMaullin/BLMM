import os
import pandas as pd
import nibabel as nib
import numpy as np

# This is a small function to load in a file based on it's prefix.
def blmm_load(filepath):

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

# This is a small function to help evaluate a string containing
# a contrast vector
def blmm_eval(c):

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


def addBlockToNifti(fname, block, blockInds,dim=None,volc=None,aff=None,hdr=None):

    # Check volc is correct datatype
    if volc is not None:

        volc = int(volc)

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
    if volc is None:

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
        data = data[:,volc].reshape((n_vox,1))

        # Add block
        data[blockInds,:] = block.reshape(data[blockInds,:].shape)
        
        # Put in the volume
        data_out[:,:,:,volc] = data[:,0].reshape(int(dim[0]),
                                                 int(dim[1]),
                                                 int(dim[2]))


    # Make NIFTI
    nifti = nib.Nifti1Image(data_out, affine, header=hdr)
    
    # Save NIFTI
    nib.save(nifti, fname)

    del nifti, fname, data_out, affine
