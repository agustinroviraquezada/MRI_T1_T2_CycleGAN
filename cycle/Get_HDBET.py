import requests
import io
import zipfile
import os
import subprocess
import shutil


def Check_HDBET(parent_dir):


  if os.path.exists(os.path.join(parent_dir,'HD_BET')) and os.path.isdir(os.path.join(parent_dir,'HD_BET')):
    print(f"The folder exists.")
  else:
      
    # URL of the repository
    url = 'https://github.com/MIC-DKFZ/HD-BET/archive/refs/heads/master.zip'

    # Send a GET request to the URL and get the response
    response = requests.get(url)

    # Set the file path to the parent directory of the current script
    #parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))

    # Get the contents of the response as a bytes-like object
    contents = io.BytesIO(response.content)

    # Extract the contents of the ZIP file to a directory named 'HD-BET-master'
    with zipfile.ZipFile(contents, 'r') as zip_file:
        zip_file.extractall(path=parent_dir)

    os.rename(os.path.join(parent_dir,"HD-BET-master"), os.path.join(parent_dir,'HD_BET'))
      
    # Set the path to the package directory
    package_dir = '/content/drive/MyDrive/TFM_2/HD_BET/.'

    # Install the package in editable mode using pip
    subprocess.check_call(['pip', 'install', '-e', package_dir])


    # Get the path to the HD-BET directory
    hd_bet_dir = "/content/drive/MyDrive/TFM_2/HD_BET/HD_BET"
    file_path = "/content/drive/MyDrive/TFM_2/HD_BET"
    url = "https://zenodo.org/record/2540681/files/HD-BET_weights.h5"


    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)
    

    # Move all files and subdirectories from the HD-BET directory to the HD_BET folder
    for item in os.listdir(hd_bet_dir):
      item_path = os.path.join(hd_bet_dir, item)
      if os.path.isfile(item_path):
        shutil.move(item_path, file_path)
      elif os.path.isdir(item_path):
        shutil.move(item_path, os.path.join(file_path, item))


  
