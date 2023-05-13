import os
import random
import os.path
import numpy as np
import torch

def seed_everything(seed=478):
  """
  Seed the random number generators for reproducibility.

  @Description:
    The seed_everything function sets the seed value for various random number generators to achieve reproducibility in a program. It sets the seed value for the random, numpy, torch CPU, and torch CUDA random number generators.

  @Input:
    - seed (int, optional): The seed value to set for the random number generators. Defaults to 478.

  @Output:
    - int: The seed value that was set for the random number generators.
  """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return seed
