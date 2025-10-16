from .post_processing_SymbR import post_processing_SymbR

def post_processing(equations, method='SymbR', params=None):
    """
    Manager to select post-processing method.
    method: 'SymbReg'
    params: dict with parameters
    """
    if method == 'SymbR':
        return post_processing_SymbR(equations, params=params)
    #elif method == 'NN':
    #     return simulate_NN(equations, params=params)
    else:
        raise ValueError(f"Unknown post-processing method '{method}'")
        
        

