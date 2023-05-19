from  HD_BET.run import run_hd_bet
import os
import requests
import re
import gzip
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import random
import glob
import cv2
import torch
from torchvision.transforms.functional import normalize
import shutil


############################ Class 1 ############################
class HDBETProcessor:
  """
  @Description: 
    A class for processing NIfTI images with HDBET (HD-BET: High-Definition Boundary Delineation)
    by applying an optional mask to the images.
  
  @Inputs:
    file_input (str): The input NIfTI file path.
    output_folder (str): The folder path where the processed NIfTI file will be saved.
    mask (str, optional): The path to the mask NIfTI file. If provided, the mask will be
    applied to the input image. Default is None.
  """

  def __init__(self, file_input, output_folder,mask=None):
    self.file_input = file_input
    self.output_folder = output_folder

    if mask:
      self.apply_mask_to_files(mask)
    else:
      self.process_nii()

  def process_nii(self):
    """
    @Description: 
      Process the input NIfTI file using HDBET and save the result in the specified output folder.
    """
    
    output_file = os.path.join(self.output_folder, os.path.basename(self.file_input))
    run_hd_bet(self.file_input, output_file)

  def apply_mask_to_files(self, mask_file):
    """
    @Description: Apply the specified mask to the input NIfTI file and save 
      the masked image in the specified output folder.

    @Inputs:
      mask_file (str): The path to the mask NIfTI file.
    """
    mask_itk = sitk.ReadImage(mask_file)
    mask_npy = sitk.GetArrayFromImage(mask_itk)

    img_itk = sitk.ReadImage(self.file_input)
    img_npy = sitk.GetArrayFromImage(img_itk)

    img_npy[mask_npy == 0] = 0

    out_itk = sitk.GetImageFromArray(img_npy)
    out_itk.CopyInformation(img_itk)

    output_file = os.path.join(self.output_folder, os.path.basename(self.file_input))
    sitk.WriteImage(out_itk, output_file)




############################ Class 2 ############################
class TransformImage():
  def __init__(self,file_path,group,save_path,target_img=(256,256),idx=None):
    """
      Initialize the TransformImage class to preprocess and save the NIfTI image slices as PyTorch tensors.

      @Description
          The TransformImage class processes a NIfTI image by extracting slices, resizing, converting them
          to PyTorch tensors, and normalizing the intensity values. The preprocessed image slices are then
          saved as .pt files in the specified path.

      @Inputs
          - file_path (str): The file path of the NIfTI image.
          - group (str): The group label to be added to the saved file name, e.g., 'D3'.
          - save_path (str): The path to save the preprocessed image slices as .pt files.
          - target_img(tupla) = (256,256): The final image size
          - idx (List), Whether if there is index to filter the .nii or not
    """


    self.group=group
    self.save_path=save_path
    self.file_path=file_path
    self.target_img=target_img

    #Apply the whole process once is called 
    self.idx=self.process_nii(indice=idx)



  def process_nii(self,indice=None):
    """
    @Description
      Preprocess the NIfTI image by extracting, resizing, and normalizing its slices.

    @Inputs
      - file_path (str): The file path of the NIfTI image.
      - idx (list): index to

    @output
      - Processes the NIfTI image by calling several transforms and saves each slice as .pt
    """

    # Load the NIfTI file
    nii_object = nib.load(self.file_path)

    #Get info about the file
    subject_match = re.search(r'sub-(\d+)', self.file_path)
    modality_match = re.search(r'_(\w+)\.', self.file_path)
    if subject_match and modality_match:
      self.subject_number = int(subject_match.group(1))
      self.modality = modality_match.group(1)
    
    # Get the 3D data array from the NIfTI object
    data = nii_object.get_fdata()

    #Select slices from the 3D np.array image
    filt_data,idx=self.FilterImg(data,indice=indice)

    #Crop image and resize to 256x25
    crop_data=self.crop_images(filt_data,margin=5,target_shape=self.target_img)

    if  crop_data.shape == (0,):
      print("To this control I am not saving anything")
    else:
      #Re-scale, convert tensor and save
      self.ApplyOperations(crop_data,idx)
    
    return idx
    
  def FilterImg(self,data,indice=None,blackper=0.6):
    """
    @Description
      Filters the input 3D numpy array by selecting images with less than 60% black pixels.

      This function first reorders the axes of the input data, then calculates the percentage
      of black pixels (0s) in each image, and selects those with less than 60% black pixels.
      Finally, it returns the filtered data along with the corresponding indices.

    @Inputs
      data (numpy.ndarray): A 3D numpy array, where the first dimension represents the number of images.
      blackper(float): The percentage of Black pixels that maximun a image has
      indide (list): A list of indices corresponding to the selected images in the input data.

    @output:
      filt_data (numpy.ndarray): A 3D numpy array containing the filtered images.
      idx (list): A list of indices corresponding to the selected images in the input data.
    """

    #Move axis
    data=np.moveaxis(data, 1, 0)

    if not indice:
      #Compute the size per image
      siz=data.shape[1]*data.shape[2]
      
      #Compute the idx fo filter and filter
      select=np.sum(data == 0, axis=(1, 2))/siz < blackper
      idx=[index for index, value in enumerate(select) if value]
      filt_data=data[idx,:,:]
    else:
      filt_data=data[indice,:,:]
      idx=indice

    return filt_data,idx

  def crop_images(self,data,margin=5,target_shape=(256, 256)):
    """
    @Description: Crop 2D images in a 3D numpy array with equal black pixel margins on all sides and resize them to the target shape.

    @Inputs:
      - data: A 3D numpy array with shape (num_images, height, width) containing the 2D images to be cropped.
      - margin: The number of black pixel margins to be left on all sides of the cropped image. Default is 5.
      - target_shape: A tuple (new_height, new_width) to resize images. Default  (256, 256).

    @Outputs:
      - crop: A 3D numpy array with shape (num_images, new_height, new_width) containing the cropped and resized images.
    """
    crop = []
    
    #Use a mask to find black margins and compute margings per each image
    for img in data:
      mask = img != 0
      row_mask = np.any(mask, axis=1)
      col_mask = np.any(mask, axis=0)

      rmin = np.maximum(np.argmax(row_mask) - margin, 0)
      rmax = row_mask.size - np.argmax(row_mask[::-1]) + margin
      cmin = np.maximum(np.argmax(col_mask) - margin, 0)
      cmax = col_mask.size - np.argmax(col_mask[::-1]) + margin

      rmax = min(rmax, img.shape[0] - 1)
      cmax = min(cmax, img.shape[1] - 1)

      #Crop the image
      cropped = img[rmin:rmax+1, cmin:cmax+1]
      #Resize to 256x256
      resized_img = cv2.resize(cropped, target_shape, interpolation=cv2.INTER_LINEAR)
      crop.append(resized_img)
    
    if not crop:
        print("No images to stack. Please check the input data.")
        return np.array([])

    return np.stack(crop)
    


  def ApplyOperations(self,img,seq):
    """
    @Description
      Apply rescale , tensor conversion, and normalization to the input image, and save it as a .pt file.

    @Inputs
      - img (numpy.ndarray): The 3D image slice to be preprocessed.
      - seq (int): The index of the image slice.

    @output
      - Preprocesses the input image and saves it as a .pt file in the specified path.
    """

    #re-scale image to 0-1
    images_scaled = img / np.max(img, axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Convert the numpy array to a PyTorch tensor
    img_tensor = torch.tensor(images_scaled, dtype=torch.float32).unsqueeze(1)

    # Normalize the tensor with mean=0.5 and std=0.5
    normalized_tensor = normalize(img_tensor, (0.5,), (0.5,))

    for i,s in zip(normalized_tensor,seq):
      torch.save(i,os.path.join(self.save_path,f"{self.group}_{self.subject_number:03d}_{s:03d}_{self.modality}.pt"))



############################ Class 3 ############################
class Process:
  def __init__(self, dataset_id, version, save_path, dest_path, target_img=(256,256)):
    """
      @Description
        The Import_nii class provides a way to download NIfTI files from the OpenNeuro dataset.
        It fetches the required dataset based on the dataset ID and version, filters the relevant files,
        and downloads them to the specified local path. The downloaded files are uncompressed and saved
        as .nii files.


      @Inputs
        - dataset_id: The dataset ID. This is the name of the folder where the file is saved as well
        - version: The version of the dataset.
        - save_path: The local path to save the downloaded files. Parent folder for dataset_id
        - group: Dataset group
        - save_path_tensor: Final path where to store the .pt
        - bound: Range where I am going to slice the .nii
        - dest_path: Final folder where to allocate T1 and T2. Dictionary with the 2 paths
        target_img (tupla): Get the image target image as tupla
    """

    self.dataset_id = dataset_id
    self.version = version
    self.save_path = save_path
    self.api_url = 'https://openneuro.org/crn/graphql'
    self.T1_path=dest_path["T1"]
    self.T2_path=dest_path["T2"]
    self.target_img=target_img


    # Define the GraphQL query to get the dataset snapshot files
    self.query = f'''
      {{
        snapshot(datasetId: "{self.dataset_id}", tag: "{self.version}") {{
          id
          tag
          files {{
            id
            filename
            size
            urls
          }}
        }}
      }}
      '''

    # Get the URL info from the query
    response = requests.post(self.api_url, json={'query': self.query})
    self.response_data = response.json()

    # Get for each subject the URL of the directory "anat"
    subjects = {i['filename']: i['id'] for i in self.response_data['data']['snapshot']['files'] if
                "sub" in i['filename']}
    Sub_anat = {k: self.RecQuery(v) for k, v in subjects.items()}

    # Save the files  in a dictionary such as  'sub-001_T2w.nii': URL
    toDownload = {}
    for k, v in Sub_anat.items():
        toDownload.update(self.GetURL(v))

    ToDownloadPairs=self.GetPairs(toDownload)
    for i in ToDownloadPairs:
      self.CoordinateProcess(i)

    #for i in range(1,len(toDownload)//2):
      #regex = re.compile(f'sub-{i:03d}')
      #sub_dict = dict(filter(lambda x: regex.search(x[0]), toDownload.items()))
      #self.CoordinateProcess(sub_dict)




  def RecQuery(self, id):
    """
    @Description
      Perform a recursive GraphQL query for the given file ID to find the corresponding 'anat' file.

    @Inputs
      - id (str): The file ID to perform the recursive query on.

    @output
        - file_id (str): The file ID of the corresponding 'anat' file.
    """

    query = f'''
      {{
        snapshot(datasetId: "{self.dataset_id}", tag: "{self.version}") {{
          files(tree: "{id}") {{
            id
            filename
            size
            urls
          }}
        }}
      }}
    '''

    response = requests.post(self.api_url, json={'query': query})
    response_data = response.json()
    response_data = response_data['data']['snapshot']['files']
    file_id = None

    for i in response_data:
        if 'anat' in i['filename']:
            file_id = i['id']

    return file_id

  def extract_subject_modality(self, url):
    """
    @Description
      Extract subject number and modality from the given URL.

    @Inputs
      - url (str): The URL to extract the subject number and modality from.

    @output
      - result (str): The formatted string containing subject number and modality, e.g., 'sub-001_T2w.nii'.
    """

    # Extract subject number and modality from the URL
    subject_match = re.search(r'sub-(\d+)', url)
    modality_match = re.search(r'sub-\d+_([a-zA-Z0-9]+)', url)
    result = None

    if subject_match and modality_match:
        subject_number = int(subject_match.group(1))
        subject = f'sub-{subject_number:03d}'
        modality = modality_match.group(1)
        result = f'{subject}_{modality}.nii'

    return result

  def GetURL(self, ant_id):
    """
    @Description
      Perform a GraphQL query for the given 'anat' file ID to find the corresponding NIfTI file URLs.

    @Inputs
      - ant_id (str): The 'anat' file ID.

    @output
      - url_dict (dict): A dictionary containing the formatted string of subject and modality as keys,
                        and corresponding NIfTI file URLs as values.
    """

    query = f'''
      {{
        snapshot(datasetId: "{self.dataset_id}", tag: "{self.version}") {{
          files(tree: "{ant_id}") {{
            id
            filename
            size
            urls
          }}
        }}
      }}
      '''

    response = requests.post(self.api_url, json={'query': query})
    response_data = response.json()
    response_data = response_data['data']['snapshot']['files']

    return {self.extract_subject_modality(i["urls"][0]): i["urls"][0] for i in response_data if
            ".nii.gz" in i["urls"][0]}

  def GetPairs(self,my_dict):
    # Prepare a dictionary to store temporary results
    temp_dict = {}

    # Regular expression pattern to match subject id
    pattern = re.compile(r"sub-\d+")

    for key, value in my_dict.items():
        subject_id = pattern.match(key).group()

        if subject_id in temp_dict:
            temp_dict[subject_id][key] = value
        else:
            temp_dict[subject_id] = {key: value}

    # Convert the dictionary of dictionaries to a list of dictionaries
    list_of_dicts = list(temp_dict.values())
    return list_of_dicts

  def CoordinateProcess(self, subDic):
    """
    @Description
      CoordinateProcess the NIfTI file from the given URL, uncompress it, and save it in the specified path.

    @Inputs
      - subDic (dictionary): Dictionary with file name and URL from one subject{sub-001_T1w.nii:URL,sub-001_T2w.nii:URL}

    @output
      - Downloads the NIfTI file, uncompresses it, and saves it in the specified path.
      - Removes the compressed .gz file after uncompressing it.
    """

    # Create a directory for the downloaded files
    download_directory = os.path.join(self.save_path, f'{self.dataset_id}_files')

    #Download .nii T1 and T2
    filenames=[]
    for k,v in subDic.items():
      filenames.append(self.Download(k, v, download_directory))

    #Apply HD-BET and do the transformationa
    for k in filenames:
      self.ApplyBET(k,download_directory)

    #Apply Transformations
    for k in filenames:
      local_file_path_nii = os.path.join(download_directory, k)
      if "_T1w.nii" in k:
        trans1=TransformImage(local_file_path_nii,self.dataset_id,self.T1_path,target_img=self.target_img,idx=None)
      else:
        TransformImage(local_file_path_nii,self.dataset_id,self.T2_path,target_img=self.target_img,idx=trans1.idx)

      os.remove(local_file_path_nii)

   
   
  
  def Download(self, filename, url, download_directory):
    """
    @Description
      Download the NIfTI file from the given URL, uncompress it, and save it in the specified path.

    @Inputs
      - filename (str): The filename of the NIfTI file to be saved.
      - url (str): The URL to download the NIfTI file from.

    @output
      - Downloads the NIfTI file
    """
    match = re.match(r'(sub-\d+)', filename)
    if match:
      sub_number = match.group(1)
      if 'T1' in url:
        filename = f"{sub_number}_T1w.nii"
      if 'T2' in url:
        filename = f"{sub_number}_T2w.nii"

    local_file_path_nii = os.path.join(download_directory, filename)
    local_file_path = local_file_path_nii + '.gz'

    response = requests.get(url)

    with open(local_file_path, 'wb') as f:
        f.write(response.content)


    # Uncompress the .gz file and save it as a .nii file
    with gzip.open(local_file_path, 'rb') as f_in:
        with open(local_file_path_nii, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the .gz file
    os.remove(local_file_path)
    print(f'Downloaded and  Uncompressed {filename} to {local_file_path_nii}')
    return filename

  def ApplyBET(self,filename,download_directory):
    """
    @Description
      Apply HD-BET and delete the mask

    @Inputs
      - filename (str): The filename of the NIfTI file
      - download_directory (str): Path where to saved it

    @output
      - .nii once the BET was applied
    """
    
    local_file_path_nii = os.path.join(download_directory, filename)

    if "_T1w.nii" in filename:
      #Apply the HD-BET
      HDBETProcessor(local_file_path_nii, download_directory)

    else:
      #Find Mask
      pattern = '*mask.nii.gz'
      mask = glob.glob(os.path.join(download_directory, pattern))[0]

      #Apply the HD-BET
      HDBETProcessor(local_file_path_nii, download_directory,mask=mask)

      #Remove mask
      os.remove(mask)

    print(f'applied HD-BET {filename}')



class  DownloadData:
  """
      Initialize the DownloadData object. Downloads and processes data for analysis

      @Description:
          Initializes the DownloadData object with the provided parameters.

      @Input:
          - save_path_nii (str): Path to the directory to save NIfTI files.
          - Datasets (dict): Dictionary of dataset IDs and versions.
          - target_img (str): Target image type.
          - dataPath (str, optional): Path to the data directory. If not provided, it uses the `save_path_nii` as the data directory.
  """

  def __init__(self,save_path_nii,Datasets,target_img,dataPath=None):

    self.save_path_nii=save_path_nii
    self.target_img=target_img

    #Define paths for T1 and T2
    if dataPath:
      self.save_path_tensor={"T1":os.path.join(dataPath,"T1") ,"T2":os.path.join(dataPath,"T2")}
    else:
      self.save_path_tensor={"T1":os.path.join(self.save_path_nii,"T1") ,"T2":os.path.join(self.save_path_nii,"T2")}

    #Create folders based on T1 and T2 paths
    for _,v in self.save_path_tensor.items():
      os.makedirs(v, exist_ok=True)

    for k,v in Datasets.items():
      if not os.path.exists(os.path.join(self.save_path_nii, f'{k}_files')):
        os.makedirs(os.path.join(self.save_path_nii, f'{k}_files'), exist_ok=True)
      
      #Details of the dataset from OpenNeuro. id:ds002330
      dataset_id=k
      version=v

      #Download data
      Process(dataset_id, version, self.save_path_nii, self.save_path_tensor,target_img=self.target_img)
      shutil.rmtree(os.path.join(self.save_path_nii, f'{k}_files'))

 


class Sanity_Check:
  def __init__(self,paths):
    self.paths=paths
    
    #Check T1 folder
    T1 = [f for f in os.listdir(paths["T1"]) if f.endswith('.pt')]
    T2 = [f for f in os.listdir(paths["T2"]) if f.endswith('.pt')]

    print(f"check if T1 length:{len(T1)} and T2 length:{len(T2)} have correspondency")

    T2_cor=[re.sub(r'(T1)(w?)', r'T2\2', f) for f in T1]
    T1_cor=[re.sub(r'(T2)(w?)', r'T1\2', f) for f in T2]

    files_only_in_list1 = set(T1) - set(T1_cor)
    files_only_in_list2 = set(T2) - set(T2_cor)

    if files_only_in_list1:
      for file_id in files_only_in_list1:
        file_path = os.path.join(paths["T1"], file_id)
        if os.path.isfile(file_path):
            print(f"Deleted {file_path}")
            os.remove(file_path)

    if files_only_in_list2:
      for file_id in files_only_in_list2:
        file_path = os.path.join(paths["T2"], file_id)
        if os.path.isfile(file_path):
            print(f"Deleted {file_path}")
            os.remove(file_path)
    
    print("Check ok")

  def count_slices_per_subject(self):
    """
        Count the number of slices per subject.

        @Description:
            Counts the number of slices per subject based on the file names.

        @Output:
            - df (pd.DataFrame): DataFrame with the slice counts per subject.
    """
    folder_path=self.paths["T1"]
    slice_counts = {}
    pattern =  r'ds\d+_\d+_(\d+)_T1w.pt'
    files=[f for f in os.listdir(folder_path) if f.endswith('.pt')]
    for filename in files:
        match = re.search(pattern, filename)
        if match:
            subject_group = match.group()
            group_folder = subject_group.split('_')[0]
            subject = subject_group.split('_')[1]
            slice_counts.setdefault(group_folder, {}).setdefault(subject, 0)
            slice_counts[group_folder][subject] += 1

    df = pd.DataFrame.from_dict(slice_counts)
    return df

  def plot_pdf(self,percentage=0.4):
    """
        Plot the probability density function (PDF) of T1 and T2 data.

        @Description:
            Plots the PDF of randomly sampled T1 and T2 data.

        @Input:
            - percentage (float, optional): Percentage of files to sample. Default is 0.4.
    """
    paths=self.paths
    T1_files    = [os.path.join(paths["T1"],f) for f in os.listdir(paths["T1"]) if f.endswith('.pt')]
    T2_files    = [os.path.join(paths["T2"],f) for f in os.listdir(paths["T2"]) if f.endswith('.pt')]
    
    sample_size = int(len(T1_files)*percentage)
    T1_sample   = random.sample(T1_files, sample_size)
    T2_sample   = random.sample(T2_files, sample_size)
    
    num_bins = 100

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for T1,T2 in zip(T1_sample,T2_sample):

      T1_array=torch.load(T1).numpy().flatten()
      T2_array=torch.load(T2).numpy().flatten()

      axs[0].hist(T1_array, num_bins, density=True, histtype='step', facecolor='g',alpha=0.05)
      axs[1].hist(T2_array, num_bins, density=True, histtype='step', facecolor='g',alpha=0.05)

    axs[0].set_xlim(-0.95, 1)
    axs[1].set_xlim(-0.95, 1)
    axs[0].set_xlabel("Pixel Values")
    axs[1].set_xlabel("Pixel Values")
    axs[0].set_ylabel("Prob Value")
    axs[1].set_ylabel("Prob Value")
    axs[0].grid()
    axs[1].grid()

    axs[0].set_title("T1 Probability Density Plot")
    axs[1].set_title("T2 Probability Density Plot")

    plt.savefig("myplot.svg", format='svg')
    plt.show()
    print(f"It has taken a sample of {sample_size} files from T1 and T2")
  
  def PlotSample(self):
    """
        Plot a sample of T1 and T2 data.

        @Description:
            Plots a sample of T1 and T2 data.
    """
    paths=self.paths
    T1_files    = [os.path.join(paths["T1"],f) for f in os.listdir(paths["T1"]) if f.endswith('.pt')]
    T1_sample   = random.sample(T1_files, 1)[0]
    T2_sample   = re.sub(r'(T1)(w?)', r'T2\2', T1_sample)

    # Move tensors to CPU and convert to NumPy arrays
    T1_cpu = torch.load(T1_sample).numpy()[0,:,:]
    T2_cpu = torch.load(T2_sample).numpy()[0,:,:]

    # Plot the tensors
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    
    axs[0].imshow(T1_cpu, cmap='gray')
    axs[0].set_title(f'T1({T1_cpu.shape[0]}x{T1_cpu.shape[1]}) - Processed')
    axs[0].axis('off')

    axs[1].imshow(T2_cpu, cmap='gray')
    axs[1].set_title(f'T2({T2_cpu.shape[0]}x{T2_cpu.shape[1]})- Processed')
    axs[1].axis('off')


    plt.tight_layout()
    plt.show()
    print(f"It is shown {os.path.basename(T1_sample)} and {os.path.basename(T2_sample)}")

