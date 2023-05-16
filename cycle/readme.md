
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
- `cycle/DataMod.py`: 
- `cycle/Get_Data.py`: 
- `cycle/Get_HDBET.py`: 
- `cycle/Get_Process_Data.py`: 
- `cycle/HyperOptim.py`: 
- `cycle/Nets.py`: 
- `cycle/SetSeed.py`: 
- `cycle/Set_FolderStructure.py`: 
- `Main.ipynb`: 
