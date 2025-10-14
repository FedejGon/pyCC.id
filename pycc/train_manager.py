from .train_NN_hybrid import train_NN_hybrid
from .train_polynomial_linear import train_polynomial_linear  
from .train_polynomial import train_polynomial  
from .train_SymbR import train_SymbR  

def train(df, equations, method='NN', params=None):
    """
    Manager to select training method.
    method: 'NN' for train_NN_hybrid, 'Poly' for train_polynomial, etc.
    params: dict with hyperparameters
    """
    if method == 'NN':
        return train_NN_hybrid(df, equations, params=params)
    elif method == 'Poly':
        return train_polynomial(df, equations, params=params)
    elif method == 'Poly_linear':
        return train_polynomial_linear(df, equations, params=params)
    elif method == 'SymbR':
        return train_SymbR(df, equations, params=params)
    else:
        raise ValueError(f"Unknown training method '{method}'")

