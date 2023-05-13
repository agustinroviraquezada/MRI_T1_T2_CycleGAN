#Necesary Packages
import os
import glob
import os.path
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import random
import re


##############################  Class 0 ############################## 
class MRIImageAugmentation:
  """
    MRIImageAugmentation class applies random transformations to an image.

  @Description:
    The MRIImageAugmentation class provides methods to apply random transformations to an image. 
    The transformations include rotation, random crop, horizontal flip, and random Gaussian noise.

  @Input:
    - TransForm_parm (dict): A dictionary containing the parameters for the transformations.
    - imgsample (torch.Tensor): An input image sample used to generate random noise.

  @Output:
    - img (torch.Tensor): The augmented image after applying the random transformations.
  """

  def __init__(self, TransForm_parm,imgsample):
    """
    @Description:
      Initialize the MRIImageAugmentation class.

    @Input:
      - TransForm_parm (dict): A dictionary containing the parameters for the transformations.
      - imgsample (torch.Tensor): An input image sample used to generate random noise.
    """
    self.rotation_range = TransForm_parm ["rotation_range"]
    self.crop_size = TransForm_parm ["crop_size"]
    self.horizontal_flip_prob = TransForm_parm ["horizontal_flip_prob"]
    self.gaussian_noise_std = TransForm_parm ["gaussian_noise_std"] 
    self.image_size=TransForm_parm ["image_size"]

    #Instaciar una sola vez
    self.rdm=[True if x in random.sample(range(0, 4), random.randint(1, 4))  else False for x in range(4)]

    #Rotation
    self.angle = random.uniform(-self.rotation_range, self.rotation_range)

    #Random crop, image with
    self.left  = random.randint(0, self.image_size[0] - self.crop_size[0])
    self.top   = random.randint(0, self.image_size[1] - self.crop_size[1])

    #random noise
    self.imagenoise=torch.randn_like(imgsample)




  def __call__(self, img):
    """
    @Description:
    Apply random transformations to the input image.

    @Input:
      - img (torch.Tensor): The input image.

    @Output:
      - img (torch.Tensor): The augmented image after applying the random transformations.
    """
    if self.rdm[0] < 0.5:
        img = self.random_gaussian_noise(img)
        print("applied_noise")
    if self.rdm[1] < 0.5:
        img = self.random_rotation(img)
        print("applied_rot")
    if self.rdm[2] < 0.5:
        img = self.random_crop(img)
        print("applied_crop")
    if self.rdm[3] < self.horizontal_flip_prob:
        img = self.random_horizontal_flip(img)
        print("applied_hori")
    return img

  def random_rotation(self, img):
    """
    @Input:
      - img (torch.Tensor): The input image.

    @Output:
      - img (torch.Tensor): The image after applying the random rotation.
    """
    return F.rotate(img, self.angle)

  def random_crop(self, img):
    """
    @Input:
      - img (torch.Tensor): The input image.

    @Output:
      - img (torch.Tensor): The image after applying the random crop.
    """
    width, height = self.image_size[0],self.image_size[1]
    crop_width, crop_height = self.crop_size
    if width < crop_width or height < crop_height:
        raise ValueError("Image size is smaller than the crop size")
    crop  = F.crop(img, self.top, self.left, crop_height, crop_width)
    return crop

  def random_horizontal_flip(self, img):
    """
    @Input:
      - img (torch.Tensor): The input image.

    @Output:
      - img (torch.Tensor): The image after applying the flip.
    """
    return F.hflip(img)

  def random_gaussian_noise(self, img):
    """
    @Input:
      - img (torch.Tensor): The input image.

    @Output:
      - img (torch.Tensor): The image after applying the random gaussian noise
    """
    noise = self.imagenoise * self.gaussian_noise_std
    return img + noise


##############################  Class 1 ############################## 
class CustomDataset(Dataset):
  """
  @Description:
    The CustomDataset class represents a dataset that contains pairs of images (T1 and T2). It provides methods to retrieve the images, apply transformations, and access their paths.

  @Input:
    - T1_files (list): A list of file paths for the T1 images.
    - T2_files (list): A list of file paths for the T2 images.
    - factor (int, optional): The factor by which the dataset should be multiplied. Defaults to 1.
    - augm_prop (float, optional): The proportion of augmented samples to include in the dataset. Defaults to None.
    - TransForm_parm (dict, optional): A dictionary containing the parameters for image augmentation transformations. Defaults to None.

  @Output:
    - img_T1 (torch.Tensor): The T1 image.
    - img_T2 (torch.Tensor): The T2 image.
    - img_path_T1 (str): The file path of the T1 image.
    - img_path_T2 (str): The file path of the T2 image.
  """
  def __init__(self, T1_files, T2_files,factor=1,augm_prop=None,TransForm_parm=None):
    """
    @Input:
      - T1_files (list): A list of file paths for the T1 images.
      - T2_files (list): A list of file paths for the T2 images.
      - factor (int, optional): The factor by which the dataset should be multiplied. Defaults to 1.
      - augm_prop (float, optional): The proportion of augmented samples to include in the dataset. Defaults to None.
      - TransForm_parm (dict, optional): A dictionary containing the parameters for image augmentation transformations. Defaults to None.
    """
    self.T1_files = T1_files
    self.T2_files = T2_files
    self.factor=factor
    self.num_originals = len(T1_files)*self.factor
    self.TransForm_parm=TransForm_parm
    

    if augm_prop :
      self.num_augmented = int(self.num_originals * augm_prop)
      self.dataSize = self.num_originals + self.num_augmented
    else:
      self.dataSize=int(min(len(self.T1_files), len(self.T2_files))*self.factor)

  def __len__(self):
    """
    @Output:
      - dataSize (int): The length of the dataset.
    """
    return self.dataSize

  def __getitem__(self, idx):
    """
    @Description:
      Get the T1 and T2 images, apply transformations if necessary, and return them along with their file paths.

    @Input:
      - idx (int): The index of the sample to retrieve.

    @Output:
      - img_T1 (torch.Tensor): The T1 image.
      - img_T2 (torch.Tensor): The T2 image.
      - img_path_T1 (str): The file path of the T1 image.
      - img_path_T2 (str): The file path of the T2 image.
    """

    if idx < self.num_originals:

      #Get Image Paths
      img_path_T1 = self.T1_files[idx]
      img_path_T2 = re.sub(r'(T1)(w?)', r'T2\2', img_path_T1)

      #Load
      img_T1 = torch.load(img_path_T1)
      img_T2 = torch.load(img_path_T2)

    else:
      idx_n=idx-self.num_originals*(idx//self.num_originals)
      
      #Get Image Paths
      img_path_T1 = self.T1_files[idx_n]
      img_path_T2 = re.sub(r'(T1)(w?)', r'T2\2', img_path_T1)

      #Load
      imgT1 = torch.load(img_path_T1)
      imgT2 = torch.load(img_path_T2)
      
      aug=MRIImageAugmentation(self.TransForm_parm,imgT2)
      transform = transforms.Compose([aug])

      img_T1 = transform(imgT1)
      img_T2 = transform(imgT2)

    return img_T1, img_T2,os.path.basename(img_path_T1),os.path.basename(img_path_T2)




##############################  Class 2 ############################## 
class CycleGANDataModule(pl.LightningDataModule):
  """
  @Description:
    The CycleGANDataModule class provides functionality to prepare and set up the data for training, validation, and testing the CycleGAN model.

  @Input:
    - paths (dict): A dictionary containing the paths for the T1 and T2 images.
    - batch_size (int, optional): The batch size for the data loaders. Defaults to 1.
    - num_workers (int, optional): The number of workers for the data loaders. Defaults to 4.
    - train_ratio (float, optional): The ratio of training data to total data. Defaults to 0.7.
    - test_size (float, optional): The ratio of test data to total data. Defaults to 0.1.
    - val_size (float, optional): The ratio of validation data to training data. Defaults to 0.2.
    - seed (int, optional): The random seed. Defaults to 478.
    - factor (int, optional): The factor by which the dataset should be multiplied. Defaults to 1.
    - augm_prop (float, optional): The proportion of augmented samples to include in the dataset. Defaults to None.
    - TransForm_parm (dict, optional): A dictionary containing the parameters for image augmentation transformations. Defaults to None.

  @Output:
    - DataLoader: The training data loader.
    - DataLoader: The validation data loader.
    - DataLoader: The test data loader.
  """

    
  def __init__(self, paths,
    batch_size=1,
    num_workers=4,
    train_ratio=0.7,
    test_size=0.1,
    val_size=0.2,
    seed=478,
    factor=1,
    augm_prop=None,
    TransForm_parm=None):


    super().__init__()
    self.T1_dir = paths["T1"]
    self.T2_dir =  paths["T2"]
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.train_ratio = train_ratio
    self.test_size = test_size
    self.val_size = val_size
    self.seed=seed
    self.factor=factor
    self.TransForm_parm=TransForm_parm
    self.augm_prop=augm_prop
    self.save_hyperparameters()



  def prepare_data(self):
    pass

  def setup(self, stage=None):
    #Get shuffled list of dirs in T1 and T2
    T1_file = [os.path.join(self.T1_dir, f) for f in os.listdir(self.T1_dir) if f.endswith('.pt')]
    T2_file = [os.path.join(self.T2_dir, f) for f in os.listdir(self.T2_dir) if f.endswith('.pt')]

    #Split to validation, training and test
    train_T1, test_T1, train_T2, test_T2 = train_test_split(T1_file,T2_file,test_size=self.test_size ,shuffle=True,random_state=self.seed)
    train_T1, val_T1, train_T2, val_T2 = train_test_split(train_T1,train_T2,test_size=(self.val_size / (1-self.test_size)), random_state=self.seed)
    
    self.trainSize=len(train_T1)
    self.valSize=len(val_T2)
    self.testSize=len(test_T2)

    self.train_dataset = CustomDataset(train_T1, train_T2,factor=self.factor,augm_prop=self.augm_prop,TransForm_parm=self.TransForm_parm)
    self.val_dataset = CustomDataset(val_T1, val_T2,factor=self.factor)
    self.test_dataset = CustomDataset(test_T1, test_T2,factor=self.factor)

    def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

  

  
