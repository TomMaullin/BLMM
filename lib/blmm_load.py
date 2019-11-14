import os
import pandas as pd
import nibabel as nib
import numpy as np

# This is a small function to load in a file based on it's prefix.
def blm_load(filepath):

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

