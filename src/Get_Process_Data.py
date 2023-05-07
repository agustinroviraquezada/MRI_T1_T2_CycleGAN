from  HD_BET.run import run_hd_bet
import os
import requests
import re
import gzip
import shutil
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import glob
import cv2
import torch
from torchvision.transforms.functional import normalize



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
    if not os.path.exists(self.output_folder):
      os.makedirs(self.output_folder, exist_ok=True)

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

    for i in range(1,len(toDownload)//2):
      regex = re.compile(f'sub-{i:03d}')
      sub_dict = dict(filter(lambda x: regex.search(x[0]), toDownload.items()))
      self.CoordinateProcess(sub_dict)




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
    os.makedirs(download_directory, exist_ok=True)

    #Download .nii T1 and T2
    for k,v in subDic.items():
      self.Download(k, v, download_directory)

    #Apply HD-BET and do the transformationa
    for k,v in subDic.items():
      self.ApplyBET(k,download_directory)

    #Apply Transformations
    for k,v in subDic.items():
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













