import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
import os
from tqdm import tqdm


class create_gif():
  """
  Create a GIF from a sequence of images and intensity distribution plots.

  @Description:
    This class creates a GIF by combining images and intensity distribution plots for each epoch of a given modality.
    The GIF displays three images (T2 Real, T2 Fake, T2 Cycle) and their corresponding intensity distribution plots.
    The images and plots are organized in a 2x3 grid layout.

  @Inputs:
    - image_path (str): The path to the folder containing the image files.
    - output_gif (str): The output filename for the GIF.
    - modality (str): The modality of the images (either "T1" or "T2").

  @Outputs:
    - output_gif (str): The path to the generated GIF file.

  """
  def __init__(self,image_path, output_gif,modality="T2"):
    """
    @Inputs:
      - image_path (str): The path to the folder containing the image files.
      - output_gif (str): The output filename for the GIF.
      - modality (str): The modality of the images (either "T1" or "T2").
    """

    self.image_path=image_path
    self.output_gif=output_gif

    # Define the image filenames and labels
    if modality=="T2":
      self.filenames = ['real_T2_epoch', 'f_T2_epoch', 'C_T2_epoch']
      self.labels = ['T2 Real', 'T2 Fake', 'T2 Cycle']
    elif modality== "T1":
      self.filenames = ['real_T1_epoch', 'f_T1_epoch', 'C_T1_epoch']
      self.labels = ['T1 Real', 'T1 Fake', 'T1 Cycle']
    else:
      print("please choose right modality T1 or T2")
    
    #Get epochs
    self.epochs = sorted(set([int(filename.split('_epoch')[1].split('.')[0]) for filename in os.listdir("/content/drive/MyDrive/Model/GIF") if 'epoch' in filename]))
    ImageLoop=[self.createImage(e) for e in tqdm(self.epochs)]

    # Save the images as a GIF using imageio
    imageio.mimsave(output_gif, ImageLoop, duration=0.2)


  def GetTensor(self,epoch):
    return [torch.squeeze(torch.load(os.path.join(image_path,f"{f}{epoch:02d}.pt"))).cpu() for f in self.filenames]

  def createImage(self,epoch):

    #Load images
    images=self.GetTensor(epoch)
    #Create figure with grid
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(10, 8))
    
    ## Real_T2
    ax1 = fig.add_subplot(gs[0, 0])# Real_T2
    ax1.imshow(images[0], cmap='gray')
    ax1.set_title(f'{self.labels[0]} - epoch:{epoch:02d}')

    ## Fake_T2
    ax2 = fig.add_subplot(gs[0, 1])# Fake_T2
    ax2.imshow(images[1], cmap='gray')
    ax2.set_title(f'{self.labels[1]} - epoch:{epoch:02d}')

    ## Cycle_T2
    ax3 = fig.add_subplot(gs[0, 2])# Cycle_T2
    ax3.imshow(images[2], cmap='gray')
    ax3.set_title(f'{self.labels[2]} - epoch:{epoch:02d}')

    # Histogram_T2
    ax4 = fig.add_subplot(gs[1, 0:])
    for im,lab,c in zip(images,self.labels,['r','b','g']):
      ax4.hist(im.flatten(), bins=100, density=True, histtype='step',facecolor=c,range=(-1,1),label=lab)
      ax4.set_xlim(-0.95,1)
      ax4.set_ylim(0,4)
      ax4.set_title('Intensity Distribution')
      ax4.set_xlabel('Intensity')
      ax4.set_ylabel('Frequency')
      ax4.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as an image (e.g., PNG)
    plt.savefig('temp.png')
    completeImage=imageio.imread('temp.png')
    plt.clf()
    return completeImage
