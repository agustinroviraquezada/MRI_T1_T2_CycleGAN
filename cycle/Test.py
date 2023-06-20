import nibabel as nib
import matplotlib.pyplot as plt
import glob
import cv2
from torchvision.transforms.functional import normalize
import re
from torch.utils.data import Dataset, DataLoader
import torch
import os
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
import pandas as pd
import numpy as np
from cycle.CycleGAN import CycleGAN
from cycle.DataMod import CycleGANDataModule,ImagePairTestSet
import random
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
from tqdm import tqdm
from  HD_BET.run import run_hd_bet
import argparse
import sys
random.seed(478)

##################### This apply to .nii #####################

class PreProssOnNii():
  def __init__(self,input,HFBET,filters,dim,blackper):
    self.HFBET=HFBET
    self.filter=filters
    self.dim=dim
    self.input=input
    self.directory=os.path.dirname(input)
    self.blackper=blackper
    self.idx=[]

  def Process(self):
    #Apply HD-BET
    if self.HFBET:
      output=os.path.join(self.directory,f"hdbet_{os.path.basename(self.input)}")
      run_hd_bet(self.input, output)
      
      self.input=output

    # Path - Check file
    data = nib.load(self.input).get_fdata()
    print(f"You just have loaded a input file of {data.shape}")

    # Apply selection
    if self.filter:
      data,self.idx=self.FilterImg(data,dim=self.dim,blackper=self.blackper)
      print(f"Slice {self.idx} has been selected")
    else:
        if self.dim!=0:
          data=np.moveaxis(data, self.dim, 0)

    # Apply crop
    CropImages=self.crop_images(data,margin=5,target_shape=(128, 128))

    # Apply
    self.ApplyOperations(CropImages)

    return self.OuputDirectory


  def ApplyOperations(self,img):

    if not self.idx:
      self.idx=list(range(img.shape[0]))

    #re-scale image to 0-1
    images_scaled = img / np.max(img, axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Convert the numpy array to a PyTorch tensor
    img_tensor = torch.tensor(images_scaled, dtype=torch.float32).unsqueeze(1)

    # Normalize the tensor with mean=0.5 and std=0.5
    normalized_tensor = normalize(img_tensor, (0.5,), (0.5,))



    self.OuputDirectory=os.path.join(self.directory,"Original")
    os.makedirs(self.OuputDirectory , exist_ok=True)
    for i,s in zip(normalized_tensor,self.idx):
      torch.save(i,os.path.join(self.OuputDirectory,f"B_{s}.pt"))


  def FilterImg(self,data,dim=0,blackper=0.6):
    #Move axis
    if dim!=0:
      data=np.moveaxis(data, dim, 0)

    #Compute the size per image
    siz=data.shape[1]*data.shape[2]
    
    #Compute the idx fo filter and filter
    select=np.sum(data == 0, axis=(1, 2))/siz < blackper
    idx=[index for index, value in enumerate(select) if value]
    filt_data=data[idx,:,:]

    return filt_data,idx

  def crop_images(self,data,margin=5,target_shape=(128, 128)):
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
          resized_img = cv2.resize(cropped, target_shape, interpolation=cv2.INTER_LINEAR)
          crop.append(resized_img)

      if not crop:
          print("No images to stack. Please check the input data.")
          return np.array([])

      return np.stack(crop)




##################### This apply the model #####################
class ModelApply():
  def __init__(self,ModelPath):
  
    params = {'lr'            : 0.0005132, #0.0002 
              'lbc_T1'        : 9.377, #7
              'lbc_T2'        : 8.834,  #8
              'lbi'           : 0.08638, #0.1
              'b1'            : 0.4984, # 0.5
              'b2'            : 0.9636, #0.999
              'batch_size'    : 1,
              'im_channel'    : 1,
              'n_epochs'      : 9000,     #When it start. High number to not apply this
              'n_epochs_decay': 9000,     #Every each epoch to do High number to not apply this
              'mode'          : "linear",
              "target_shape"  : 1,
              "resnet_neck"   : 7,      #6
              "features"      : 56}

    model=CycleGAN(params).load_from_checkpoint(checkpoint_path=ModelPath)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval() 
    self.model=model.to(self.device)  


  def procesar(self,image_directory,modality,directory):
    datalist=[]
    image=glob.glob(f'{image_directory}/*')

    targetPath=os.path.join(directory,"Synthetic")
    os.makedirs(targetPath, exist_ok=True)

    if (isinstance(image, list)) and (len(image)>1):

      if modality=="T1":
        for img in image:
          tensor_image=torch.load(img).to(self.device)
          with torch.no_grad():
            datalist.append(self.model.G_T1_T2(tensor_image))

          for d in datalist:
            torch.save(d,os.path.join(targetPath,f"s_{os.path.basename(img)}"))
      
      elif modality=="T2":
        for img in image:
          tensor_image=torch.load(img).to(self.device)
          with torch.no_grad():
            datalist.append(self.model.G_T2_T1(tensor_image))

          for d in datalist:
            torch.save(d,os.path.join(targetPath,f"s_{os.path.basename(img)}"))

            


    else:
      if modality=="T1":
        tensor_image=torch.load(image[0]).to(self.device)
        with torch.no_grad():
          output=self.model.G_T1_T2(tensor_image)

        torch.save(output,os.path.join(targetPath,os.path.basename(image[0])))

      elif modality=="T2":
        tensor_image=torch.load(image[0]).to(self.device)
        with torch.no_grad():
          output=self.model.G_T2_T1(tensor_image)

        torch.save(output,os.path.join(targetPath,os.path.basename(image[0])))




##################### This apply operations on single slice #####################

class SliceOperation():
  def __init__(self,inputs,crop=False,operation=False):
    '''
      Input= Path from a slice tensor. 1x128x128 in an ideal case
    '''

    #This expect a tensor slice, already prepared.
    self.input=inputs
    self.directory=os.path.dirname(inputs)
    self.crop=crop
    self.operation=operation


  def GetResult(self):
    data=torch.squeeze(torch.load(self.input).to("cpu")).numpy()
    pathname=os.path.join(os.path.dirname(self.input),"Original")
    #Procesar
    if self.crop:
      data=self.cropslice(data)

    if self.operation:
      datapath=self.SliceOperations(data)
      pathname=datapath
    else:
      os.makedirs(pathname,exist_ok = True)
      file_name, file_extension = os.path.splitext(self.input)
      savename=os.path.join(pathname,f"{os.path.basename(file_name)}.pt")
      torch.save(torch.load(self.input),savename)

    return pathname
    
  def cropslice(self, data, margin=5, target_shape=(128, 128)):
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    mask = data != 0
    row_mask = np.any(mask, axis=0)
    col_mask = np.any(mask, axis=1)
    
    rmin = np.maximum(np.argmax(row_mask) - margin,0)
    rmax = row_mask.size - np.argmax(row_mask[::-1]) + margin
    cmin = np.maximum(np.argmax(col_mask) - margin,0)
    cmax = col_mask.size - np.argmax(col_mask[::-1]) + margin

    
    # Crop the image
    cropped = data[cmin:cmax,rmin:rmax]


    # Resize to target shape
    resized_img = cv2.resize(cropped, target_shape, interpolation=cv2.INTER_LINEAR)

    return np.array(resized_img)
    

  def SliceOperations(self,img):
    #re-scale image to 0-1
    images_scaled  = (img-np.min(img)) / (np.max(img)-np.min(img))

    # Convert the numpy array to a PyTorch tensor
    img_tensor = torch.tensor(images_scaled, dtype=torch.float32)[None,:,:]


    # Normalize the tensor with mean=0.5 and std=0.5
    normalized_tensor = normalize(img_tensor, (0.5,), (0.5,))
    filesaved=os.path.join(self.directory,"Process_Image")
    torch.save(normalized_tensor,filesaved)
    return filesaved



if __name__ == '__main__':
  # Definir el argumento de línea de comandos
  parser = argparse.ArgumentParser(description='Pipeline de procesamiento')
  parser.add_argument('--input', required=True , type=str,  help='Input file path. File must be a .nii or tensor 128 x 128')
  parser.add_argument('--model', required=True , type=str, help='Path to the model')
  parser.add_argument('--Modality', required=True , type=str, help='Input Modality')
  parser.add_argument('--HDBET', required=False , type=bool, help='Apply HDBET only for .nii')
  parser.add_argument('--filters', required=False , type=int, help='Apply image selection based on the black pixel proportion')
  parser.add_argument('--dim', required=False , type=int, help='Dimention where to iterate. For .nii where is the dimention to iterate')
  parser.add_argument('--crop', required=False , type=bool, help='Apply centering crop')
  parser.add_argument('--operation', required=False , type=bool, help='Apply Image normalization')
  parser.add_argument('--BlackProportion', required=False , type=float, help='Black Pixel Proportion threshold')


  # Print help if no arguments are provided
  if len(sys.argv) == 1:
      parser.print_help(sys.stderr)
      sys.exit(1)

  # Obtener los argumentos de la línea de comandos
  args = parser.parse_args()

  #parameter
  file_name, file_extension = os.path.splitext(args.input)
  model=ModelApply(args.model)
  Modality=args.Modality


  #Check inputs and pre-processing
  if file_extension == ".pt":
    operation=args.operation
    crop=args.crop
    im=args.input
    OuputDirectory=SliceOperation(im,crop=crop,operation=operation).GetResult()
    model.procesar(OuputDirectory,Modality,os.path.dirname(OuputDirectory))
    

  elif file_extension == ".nii" or file_extension == ".nii.gz" or file_extension == ".gz":
    
    HDBET=args.HDBET
    filt=bool(args.filters)
    dim=args.dim
    im=args.input
    blackper=args.BlackProportion
    OuputDirectory=PreProssOnNii(im,HDBET,filt,dim,blackper).Process()
    model.procesar(OuputDirectory,Modality,os.path.dirname(OuputDirectory))
   

  else:
    print("Please check the help or the code")
