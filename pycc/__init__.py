# src/__init__.py

# External libraries
import numpy as np
import torch

# Define initial seed to have repetitivity 
np.random.seed(0) 
torch.manual_seed(10) 


from .train_manager import train
#from .train_NN_hybrid import train_NN_hybrid
#from .train_polynomial import train_polynomial
#from .train_polynomial_linear import train_polynomial_linear
#from .train_SymbReg import train_SymbReg

from .simulate_manager import simulate
#from .simulate_th import simulate_th


from .process_evals_SymbR import process_evals_SymbR

from .eval_cc import eval_cc


__version__ = "0.0"
__all__ = ["train"]
