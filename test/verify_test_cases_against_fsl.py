import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import glob
import os
import nibabel as nib
import numpy as np

def main(fsl_folder, blm_folder):

    # Check Betas
    fsl_file = os.path.join(fsl_folder, 'fsl_vox_betas.nii.gz')
    blm_file = os.path.join(blm_folder, 'blm_vox_con.nii')

    for i in range(0,3):
        
        # Load in FSL files
        fsl_dat = nib.load(fsl_file).get_data()
        fsl_dat = fsl_dat[:,:,:,i]

        # Load in BLM files
        blm_dat = nib.load(blm_file).get_data()
        blm_dat = blm_dat[:,:,:,i]

        # Check how close blm and fsl are voxelwise
        compar=np.isclose(blm_dat, fsl_dat)

        # Count number of non-zero voxels agreeing in compar
        compar = compar[(blm_dat!=0)*(fsl_dat!=0)]
        nv = sum(compar)
        pv = sum(compar)/compar.shape[0]

        # Tell the user we are testing
        print('======================================================')
        print('Testing: ' + blm_file)
        print('Against: ' + fsl_file)

        # Check if values are all close to expected
        if pv>0.7:
           result = 'PASSED'
        else:
           result = 'FAILED'

        print("------------------------------------------------------")
        print("Number of agreeing non-zero voxels: " + str(nv))
        print("Percentage of agreeing non-zero voxels: " + str(pv))

        # Output result
        print('Result: ' + result)

    # Check T stats
    fsl_file = os.path.join(fsl_folder, 'fsl_vox_Tstat_c.nii.gz')
    blm_file = os.path.join(blm_folder, 'blm_vox_conT.nii')

    for i in range(0,3):
        
        # Load in FSL files
        fsl_dat = nib.load(fsl_file).get_data()
        fsl_dat = fsl_dat[:,:,:,i]

        # Load in BLM files
        blm_dat = nib.load(blm_file).get_data()
        blm_dat = blm_dat[:,:,:,i]

        # Check how close blm and fsl are voxelwise
        compar=np.isclose(blm_dat, fsl_dat)

        # Count number of non-zero voxels agreeing in compar
        compar = compar[(blm_dat!=0)*(fsl_dat!=0)]
        nv = sum(compar)
        pv = sum(compar)/compar.shape[0]

        # Tell the user we are testing
        print('======================================================')
        print('Testing: ' + blm_file)
        print('Against: ' + fsl_file)

        # Check if values are all close to expected
        if pv>0.7:
           result = 'PASSED'
        else:
           result = 'FAILED'

        print("------------------------------------------------------")
        print("Number of agreeing non-zero voxels: " + str(nv))
        print("Percentage of agreeing non-zero voxels: " + str(pv))

        # Output result
        print('Result: ' + result)

    # Check F stat
    fsl_file = os.path.join(fsl_folder, 'fsl_vox_Fstat_c.nii.gz')
    blm_file = os.path.join(blm_folder, 'blm_vox_conF.nii')
        
    # Load in FSL files
    fsl_dat = nib.load(fsl_file).get_data()

    # Load in BLM files
    blm_dat = nib.load(blm_file).get_data()
    fsl_dat = fsl_dat.reshape(blm_dat.shape)

    # Check how close blm and fsl are voxelwise
    compar=np.isclose(blm_dat, fsl_dat)

    # Count number of non-zero voxels agreeing in compar
    compar = compar[(blm_dat!=0)*(fsl_dat!=0)]
    nv = sum(compar)
    pv = sum(compar)/compar.shape[0]

    # Tell the user we are testing
    print('======================================================')
    print('Testing: ' + blm_file)
    print('Against: ' + fsl_file)

    # Check if values are all close to expected
    if pv>0.7:
        result = 'PASSED'
    else:
        result = 'FAILED'

    print("------------------------------------------------------")
    print("Number of agreeing non-zero voxels: " + str(nv))
    print("Percentage of agreeing non-zero voxels: " + str(pv))

    # Output result
    print('Result: ' + result)

    # Check covariances
    fsl_file = os.path.join(fsl_folder, 'fsl_vox_cov_c.nii.gz')
    blm_file = os.path.join(blm_folder, 'blm_vox_conSE.nii')
    fsl_dat = fsl_dat.reshape(blm_dat.shape)

    for i in range(0,3):
        
        # Load in FSL files
        fsl_dat = nib.load(fsl_file).get_data()
        fsl_dat = fsl_dat[:,:,:,i]

        # Load in BLM files
        blm_dat = nib.load(blm_file).get_data()
        blm_dat = blm_dat[:,:,:,i]**2

        # Check how close blm and fsl are voxelwise
        compar=np.isclose(blm_dat, fsl_dat)

        # Count number of non-zero voxels agreeing in compar
        compar = compar[(blm_dat!=0)*(fsl_dat!=0)]
        nv = sum(compar)
        pv = sum(compar)/compar.shape[0]

        # Tell the user we are testing
        print('======================================================')
        print('Testing: ' + blm_file)
        print('Against: ' + fsl_file)

        # Check if values are all close to expected
        if pv>0.7:
           result = 'PASSED'
        else:
           result = 'FAILED'

        print("------------------------------------------------------")
        print("Number of agreeing non-zero voxels: " + str(nv))
        print("Percentage of agreeing non-zero voxels: " + str(pv))

        # Output result
        print('Result: ' + result)


    print('======================================================')

if __name__ == "__main__":
    main()

