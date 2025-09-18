from .simulate_th import simulate_th
#from .train_polynomial_linear import train_polynomial_linear  
#from .train_polynomial import train_polynomial  
#from .train_SymbReg import train_SymbReg  

def simulate(equations, method='Theoretical', params=None):
    """
    Manager to select simulation method.
    method: 'Theoretical', 'NN', 'Poly', 'SymbReg', etc.
    params: dict with simulation parameters (t_span, y0, noise, etc.)
    """
    if method == 'Theoretical':
        return simulate_th(equations, params=params)
    # elif method == 'NN':
    #     return simulate_NN(df, equations, params=params)
    # elif method == 'Poly':
    #     return simulate_Poly(df, equations, params=params)
    else:
        raise ValueError(f"Unknown simulation method '{method}'")
