The .nii in this example suffered transformation like. This was like this since the generation.py code has done to iterate over the first axis of the .nii
Therefore, the stack images has to be contained in the axis 1 and 2

import nibabel as nib
import numpy as np

# Load the .nii file
nii_file_t1 = nib.load('/content/BraTS20_Training_002_t1.nii')
nii_file_t2 = nib.load('/content/BraTS20_Training_002_t2.nii')

data1 = nii_file_t1.get_fdata()
modified_data1 = data1.transpose(2, 0, 1)  # Modify the axis order as desired
modified_nii1 = nib.Nifti1Image(modified_data1, nii_file_t1.affine, header=nii_file_t1.header)
nib.save(modified_nii1, '/content/BraTS20_Training_002_t1.nii')


data2 = nii_file_t2.get_fdata()
modified_data2 = data2.transpose(2, 0, 1)  # Modify the axis order as desired
modified_nii2 = nib.Nifti1Image(modified_data2, nii_file_t2.affine, header=nii_file_t2.header)
nib.save(modified_nii2, '/content/BraTS20_Training_002_t2.nii')
