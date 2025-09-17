from .train_NN_hybrid import train_NN_hybrid
from .train_polynomial import train_polynomial  
from .train_SymbReg import train_SymbReg  

def train(df, equation, method='NN', params=None):
    """
    Manager to select training method.
    method: 'NN' for train_NN_hybrid, 'Poly' for train_polynomial, etc.
    params: dict with hyperparameters
    """
    if method == 'NN':
        return train_NN_hybrid(df, equation, params=params)
    elif method == 'Poly':
        return train_polynomial(df, equation, params=params)
    elif method == 'SymbReg':
        return train_SymbReg(df, equation, params=params)
    else:
        raise ValueError(f"Unknown training method '{method}'")

