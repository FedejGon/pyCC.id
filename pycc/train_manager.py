from .train_NN_hybrid import train_NN_hybrid
from .train_polynomial_linear import train_polynomial_linear  
from .train_polynomial import train_polynomial  
from .train_SymbR import train_SymbR  
from .train_SparseR import train_SparseR
from .train_GP import train_GP

def train(df, equations, method='NN', params=None):
    """
    Trains a model to identify the system dynamics from data.

    This function acts as a dispatcher, selecting the appropriate training
    method based on the `method` argument. It passes the data, equations,
    and hyperparameters to the selected specialized training function.

    :param df: The input data.
    :type df: pandas.DataFrame
    :param equations: A string or list of strings defining the system equations.
                      It defines the equations we want to fit with a given df.
                      The user can include functions and parameters defined by f_i and a_j, respectively 
                      (i,j=0,1,...). It can also include time dependent functions, which must be
                      defined in the df. 
    :type equations: str or list[str]
    :param method: The system identification method to use.
                   Options: \'NN\', \'Poly\', \'SymbR\'. **Default: \'NN\'**. 
    :type method: str, optional
    :param params: A dictionary of hyperparameters for the training process.
                   See the notes below for method-specific parameters.
    :type params: dict, optional

    :raises ValueError: If an unknown `method` is specified.

    :return: A tuple containing the trained models, plotting results, and identified scalar parameters.
    :rtype: tuple(dict, list, dict)

    .. note::
        The `params` dictionary can contain different keys depending on the
        selected `method`. For method='NN'`:

        * **'neurons'** (*int*): Number of neurons in each hidden layer. Default: 100.
        * **'layers'** (*int*): Number of hidden layers in the neural network. Default: 3.
        * **'lr'** (*float*): Learning rate for the optimizer. Default: 1e-3.
        * **'epochs'** (*int*): Number of training epochs. Default: 1000.
        * **'device'** (*str*): The computing device ('cpu', 'cuda', 'xpu', 'automatic'). Default: 'automatic'.
        * ... and other ..
    """

#    .. seealso::
#        For a complete definition of params, refer to Documentation:
#    """
    if method == 'NN':
        return train_NN_hybrid(df, equations, params=params)
    elif method == 'Poly':
        return train_polynomial(df, equations, params=params)
    elif method == 'Poly_linear':
        return train_polynomial_linear(df, equations, params=params)
    elif method == 'SymbR':
        return train_SymbR(df, equations, params=params)
    elif method == 'SparseR':
        return train_SparseR(df, equations, params=params)
    elif method == 'GP':
        return train_GP(df, equations, params=params)
    else:
        raise ValueError(f"Unknown training method '{method}'")

