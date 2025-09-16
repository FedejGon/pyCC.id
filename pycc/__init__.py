# src/__init__.py

# External libraries
import numpy as np
import torch

# Define initial seed to have repetitivity 
np.random.seed(0) 
torch.manual_seed(10) 


from .train_manager import train
from .train_NN_hybrid import train_NN_hybrid
from .train_polynomial import train_polynomial

from .eval_cc import eval_cc


__version__ = "0.0"
__all__ = ["train"]
