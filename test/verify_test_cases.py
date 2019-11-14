import warnings as w
# These warnings are caused by numpy updates and should not be
# output.
w.simplefilter(action = 'ignore', category = FutureWarning)
import glob
import os
import nibabel as nib
import numpy as np

def main(folder, gt_folder):

    # List files to verify
    filestotest = glob.glob(os.path.join(folder, '*.nii*'))

    # Loop through files checking against ground truth
    for file in filestotest:

        # Work out which is ground truth file
        gt_file = os.path.join(gt_folder,os.path.split(file)[1])

        # Load ground truth and truth
        f_dat = nib.load(file).get_data()
        gt_f_dat = nib.load(gt_file).get_data()

        # Tell the user we are testing
        print('======================================================')
        print('Testing: ' + file)
        print('Against: ' + gt_file)

        # Check if values are all close to expected
        if np.allclose(gt_f_dat, f_dat):
            result = 'PASSED'
        else:
            result = 'FAILED'

        # Output result
        print('Result: ' + result)

    print('======================================================')




if __name__ == "__main__":
    main()
