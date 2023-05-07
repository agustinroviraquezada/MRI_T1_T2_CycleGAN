from torch import nn


class ResBlock(nn.Module):
  '''
  ResBlock Class:
  @Based on the paper: 
  - [1] Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros.
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
    in IEEE International Conference on Computer Vision (ICCV), 2017
  - [2] Dar, S. U., Yurt, M., Karacan, L., Erdem, A., Erdem, E., & Cukur, T. (2019). 
    Image synthesis in multi-contrast MRI with conditional generative adversarial networks.
    IEEE transactions on medical imaging, 38(10), 2375-2388.
  - [3] https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
    

  @Description
    This class implement a Residual block, to use later in class Generator. 
    As reference one describes, residual blocks contains 2 convolutional layer
    with a intance normalization interspersed. Finally, to achive the residual
    efect, the output is added to the input.

  @Inputs
    Ich: Input channel
    k_size. Kernel size Default 3, as it is defined in [1]
    p. Padding mode as default 1,  as it is defined in [1]
    p_m. Padding mode as 'reflect' by default,
    dropOut=None


  @Outputs
    Returns the output of the block Original input+residual   
  '''
  
  def __init__(self,Ich,k_size=3,p=1,p_m='reflect',dropOut=None):
    super(ResBlock, self).__init__()

    ######################  Define the block ######################
    self.Resblock=nn.Sequential()
    self.Resblock.add_module("conv1",nn.Conv2d(Ich,Ich,kernel_size=k_size,padding=p,padding_mode=p_m))
    self.Resblock.add_module("Inst_1",nn.InstanceNorm2d(Ich))
    self.Resblock.add_module("Relu_1",nn.ReLU())
    if dropOut: self.Resblock.add_module("Drop",nn.Dropout(dropOut))
    self.Resblock.add_module("conv2",nn.Conv2d(Ich,Ich,kernel_size=k_size,padding=p,padding_mode=p_m))
    self.Resblock.add_module("Inst_2",nn.InstanceNorm2d(Ich))

  def forward(self, x):
    '''
        x: image tensor of shape (batch size, channels, height, width)
    '''
    original_x = x.clone()
    x = self.Resblock(x)
    return original_x + x




class Generator(nn.Module):
  '''
  ResBlock Class:
  @Based on the paper: 
  - [1] Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros.
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
    in IEEE International Conference on Computer Vision (ICCV), 2017
  - [2] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020).
    Generative adversarial networks. Communications of the ACM, 63(11), 139-144.
  - [3] Dar, S. U., Yurt, M., Karacan, L., Erdem, A., Erdem, E., & Cukur, T. (2019). 
    Image synthesis in multi-contrast MRI with conditional generative adversarial networks.
    IEEE transactions on medical imaging, 38(10), 2375-2388.
  - [4] https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
    

  @Description
    Following references [1,2] a generator with 9 residual blocks consists is created
    to use the cyclegan on images of 256x26. However this is the standard model. The 
    sript is open to add more residual blocks or change input parameters such as kernel,
    padding.
    
    The default layers are:
    c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

    Notation example:
    - c7s1-k denote a7×7 Convolution-Instance Norm-ReLU layer with k filters and stride 1. 
    - dk denotes a3×3 Convolution-InstanceNorm-ReLU layer  with k filters  and stride 2.
    - Rk denotes a residual block that contains two 3×3 convolutional layers with the same number of filters on both layer.
    - uk denotes a 3×3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2.

  @Inputs
    in_f: input (image)
    out_f. by default 64 (int), number of filters in first layer. Take into account the input size and the encoder leverls
    lvl_aut: Autoencoder levels, by default 2. Since encoder and deconder has same levels, this is a int with the levels of only one of them
    lvl_resnt: Resinual neck levels,  by default 9.



  @Outputs
    Returns the output of the block Original input+residual   
  '''
  
  def __init__(self,in_f,out_f=64,lvl_aut=2,lvl_resnt=9,upsam=None):
    super(Generator, self).__init__()

    ######################  Define constants and variables ######################
    f_deep=[2**i for i in range(1,lvl_aut+1)] #level scale
    rInput=out_f*(2**lvl_aut) # Input size Rest blocks
    self.gen=nn.Sequential()

    ######################  Define the encoder ######################
    self.gen.add_module(f"c7s1_{out_f}", self.EncLayer(in_f,out_f,nor=False,k=7,p=3,s=1,act='relu'))#c7s1-64

    #d128,d256
    Ich=out_f
    for f in f_deep:
      self.gen.add_module(f"d{out_f*f}", self.EncLayer(Ich,out_f*f,nor=True, k=3,p=1,s=2,act='relu'))
      Ich=out_f*f


    ######################  Define the residual neck ######################
    for l in range(lvl_resnt):
     self.gen.add_module(f"R{rInput}_{l}", ResBlock(rInput,k_size=3,p=1,p_m='reflect',dropOut=None))  #R256 x 9


    ######################  Define the Decoder ######################
    #u128,u64
    Ich=rInput
    for f in f_deep:
     self.gen.add_module(f"u{rInput//f}", self.DecLayer(Ich,rInput//f,nor=True,k=3,s=2,p=1,op=1,act="relu"))
     Ich=rInput//f

    ######################  Last layer ######################
    #self.c7s1_3 = nn.Sequential()
    self.gen.add_module("c7s1_3", nn.Conv2d(out_f,
                              in_f, 
                              kernel_size=7, 
                              padding=3, 
                              stride=1, 
                              padding_mode='reflect'))
    
    self.gen.add_module("Tanh", nn.Tanh())


  #--------------------------------------- Methods ---------------------------------------#
  def EncLayer(self,Ich,Och,nor=True, k=3,p=1,s=2,act='relu'):
    '''
    @Description
      Creates dk layer using a convoutional layer, with ot without normalizaton.
      This is the downsampling part, in other words the layer of the encoder

    @Inputs
      Ich: Input channels, int
      Och: Output channles, int
      k. kernel size By default 3
      p. Padding, by default 1
      s. Stride, as is defined in [1] it is 2 as default
      relu. Boolean, True as defaul uses Relu. False use nn.LeakyReLU(0.2)
      drop.  by default is None. Float number indicates the percentage of a dropout layer
      norm. by default is True for instance normalization. False indicates no instance normalization layer.

    @Outputs
      Sequential model wich correspond a dk layer 
    '''
    m = nn.Sequential()
    m.add_module("conv1", nn.Conv2d(Ich, Och, kernel_size=k, padding=p, stride=s, padding_mode='reflect'))
    if nor: m.add_module("Instancenorm", nn.InstanceNorm2d(Och)) 
    m.add_module("activation", nn.ReLU()) if act else m.add_module("activation", nn.LeakyReLU(0.2))
    return m

  def DecLayer(self,Ich,Och,nor=True,k=3,s=2,p=1,op=1,act="relu"):
    '''
    @Description
      Creates uk layer using a deconvoutional layer, with or without normalization.
      This is the upsampling part, in other words the layer of the decoder

    @Inputs
      Ich: Input channels, int
      Och: Output channles, int
      k_size. kernel size By default 3
      p. Padding, by default 1
      p_m. Padding mode, by default 'reflect'
      s. Stride, as is defined in [1] it is 2 as default
      relu. Boolean, True as defaul uses Relu. False use nn.LeakyReLU(0.2)
      drop.  by default is None. Float number indicates the percentage of a dropout layer
      norm. by default is True for instance normalization. False indicates no instance normalization layer.
      upsampling. by default is None. int number indicates the scale factor for upsampling

    @Outputs
      Sequential model wich correspond a uk layer 
    '''
    m = nn.Sequential()
    m.add_module("dconv1",  nn.ConvTranspose2d(Ich,Och, kernel_size=k, stride=s, padding=p, output_padding=op))
    if nor: m.add_module("Instancenorm", nn.InstanceNorm2d(Och)) 
    m.add_module("activation", nn.ReLU()) if act else m.add_module("activation", nn.LeakyReLU(0.2))
    return m


  #---------------------------------------  Call funtion ---------------------------------------#
  def forward(self, x):
    x=self.gen(x)
    return x




class Discriminator(nn.Module):
  '''
  Discriminator Class: 
  @Based on the paper: 
   - [1] Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros.
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
    in IEEE International Conference on Computer Vision (ICCV), 2017
   - [2] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020).
    Generative adversarial networks. Communications of the ACM, 63(11), 139-144.
   - [3] https://arxiv.org/abs/1703.10593
   - [4] Dar, S. U., Yurt, M., Karacan, L., Erdem, A., Erdem, E., & Cukur, T. (2019). 
    Image synthesis in multi-contrast MRI with conditional generative adversarial networks.
    IEEE transactions on medical imaging, 38(10), 2375-2388. 
    

  @Description
    The discriminator yields  a matrix of values classifying corresponding 
    portions of the image as real or fake.

    Following references [1,2,3] for   discriminator   net-works, we use 70×70 PatchGAN.  
    After the last layer, we apply a convolution to produce a 1-dimensional output. leaky ReLUs
    with a slope of 0.2 is used to deal with vanishing problems. After the last layer, we apply
    a convolution to produce a 1-dimensional output
     
    The default layers are:
    C64-C128-C256-C512-C1

    Notation example:
    - Ck denote a 4×4 Convolution - InstanceNorm - LeakyReLU (0.2) layer with k filters and stride 2


  @Inputs
    ICh: the number of image input channels
    HCh: Hidden layers. the initial number of discriminator convolutional filters
    n. Number of layer to implement


  @Outputs
    Returns patchGAN Discriminator 
  '''

  def __init__(self,ICh,HCh=64,n=3):
    super(Discriminator, self).__init__()

    ######################  Define constants and variables ######################
    f_deep=[2**i for i in range(1,n+1)] #level scale
    self.disc=nn.Sequential()
    self.disc.add_module(f"C{HCh}",self.lcreation(ICh,HCh,k_size=4,p=1,s=2,drop=None,relu=False,norm=False))

    ######################  Define layers ######################
    input=HCh
    for f in f_deep:
      self.disc.add_module(f"C{HCh*f}",self.lcreation(input,HCh*f,k_size=4,p=1,s=2,drop=None,relu=False,norm=True))
      input=HCh*f

    ######################  Define layers ######################
    self.disc.add_module(f"C1",nn.Conv2d(HCh*(2**n), 1, kernel_size=4, padding=1))

  #---------------------------------------  Methods ---------------------------------------#
  def lcreation(self,Ich,Och,k_size=4,p=1,s=2,drop=None,relu=True,norm=True):
    '''
    @Description
    Creates layers according reference [1]. Using Convolution - InstanceNorm - LeakyReLU (0.2) 
    if it is need it.

    @Inputs
     ICh: the number of image input channels
     Och: Number of filters in the conv layer
     k_size. Kernal size, by default 4
     p. Padding by default 1
     s stride by default 2
     drop dropout factor, by default None
     relu if apply relu o leaky. By default Relu using True
     norm. Apply instance normalization. By defaul True

    @Outputs
      Returns sequential model.
    '''

    m = nn.Sequential()
    m.add_module("conv1", nn.Conv2d(Ich, 
                              Och, 
                              kernel_size=k_size, 
                              padding=p, 
                              stride=s))
    
    if norm: m.add_module("Instancenorm", nn.InstanceNorm2d(Och)) 
    m.add_module("activation", nn.ReLU()) if relu else m.add_module("activation", nn.LeakyReLU(0.2))
    if drop: m.add_module("Dropout", nn.Dropout(drop)) 
    return m
  #---------------------------------------  Call funtion ---------------------------------------#
  def forward(self, x):
    x=self.disc(x)
    return x
