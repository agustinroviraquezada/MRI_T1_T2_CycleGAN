
# CycleGAN Package
This project contains the implementation of a Cycle Generative Adversarial Network (CycleGAN) for various tasks, as package.


## Project Structure
```
├── cycle
│   ├── CallBacks.py
│   ├── CycleGAN.py
│   ├── DataMod.py
│   ├── Get_Data.py
│   ├── Get_HDBET.py
│   ├── Get_Process_Data.py
│   ├── HyperOptim.py
│   ├── Nets.py
│   ├── SetSeed.py
│   └── Set_FolderStructure.py
```

## File Descriptions

- `cycle/CallBacks.py`: CallBacks.py: This file contains two classes, CustomModelCheckpoint and CreateTensor, which extend the functionality of PyTorch Lightning's pl.Callback.

  * CustomModelCheckpoint: This class is used to save the state of a model during training. It can save the best model based on a specific metric and can also save model checkpoints periodically during training. In the on_validation_epoch_end method, it saves the best model checkpoint based on the monitored metric and in the on_train_epoch_end method, it saves a model checkpoint every N epochs.
  * CreateTensor: This class is used to save a PyTorch tensor representation of an image at the end of the first step in each validation epoch. The on_validation_batch_end method is responsible for this, saving the generated image tensor from the validation step outputs.
![UML Diagram from callbacks](https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/CallBacks.svg)



- `cycle/CycleGAN.py`: 
- `cycle/DataMod.py`: creates a pipeline for processing brain MRI images for training the deep learning model. The process involves multiple steps.

  1. Download: Fetching data from a specified URL, unzipping the .gz files, and storing them locally in NIfTI format.
  2. Brain Extraction: Applying HD-BET [(High-resolution Data-driven Brain Extraction Tool)](https://github.com/MIC-DKFZ/HD-BET)
  to remove non-brain tissue from the images.
  3. Image Transformation: Transforming the processed images, which includes filtering out black images, cropping and resizing the images,  and then saving them as PyTorch tensors for further processing.
  4. Sanity Check: Checking if the lengths of T1 and T2 images are equivalent, removing any mismatching files, and providing a count of slices per subject.

These classes include are:
 * Process: Handles the overall processing of a dataset. It fetches data from the API, downloads and extracts the NIfTI files, applies HD-BET, and transforms the images.
 * DownloadData: Handles the downloading of data from different datasets. It makes sure the correct directories are created, and initiates the processing of each dataset.
 * HDBETProcessor: Applies the HD-BET tool on the input NIfTI images. It can also apply a mask to the images if provided.
 * TransformImage: Processes the NIfTI images, filters out the black images, crops, resizes and normalizes the images, and saves them as PyTorch tensors.
 * Sanity_Check: Checks if there is a one-to-one correspondence between T1 and T2 images for each subject. It also provides the functionality to count the number of slices per subject.
![UML Diagram from Data processing](https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/Get_Process_Data.svg)



- `cycle/Get_Data.py`: 
- `cycle/Get_HDBET.py`: 
- `cycle/Get_Process_Data.py`: 
- `cycle/HyperOptim.py`: 
- `cycle/Nets.py`: 
- `cycle/SetSeed.py`: 
- `cycle/Set_FolderStructure.py`: 
- `Main.ipynb`: 
