Selected functions
==================

.. toctree::
   :maxdepth: 2


This section provides a detailed reference for the core functions in the ``pycc`` library.

pycc.train()
==========

This is the main function for identifying system dynamics from data. It acts as a manager that calls a specific training method based on the ``method`` parameter.

.. autofunction:: pycc.train
   :noindex:
 

**Example Usage:**

.. code-block:: python

    import pycc
    import pandas as pd

    # Assume we have simulated a second order system
    # and obtained x1,x2,x1_dot,x2_dot,Fext(t) variables
    # Now define a pandas DataFrame with your data
    df = pd.DataFrame({
        'x1':x1_data,
        'x2':x2_data,
        'x1_dot':x1_dot_data,
        'x2_dot':x2_dot_data,
        'F_ext': F_ext_val
    })
    
    #now define the system we want to fit
    eqs = [
        'x1_dot = x2',
        'x2_dot = F_ext - f1(x2) - f2(x1)'
    ]

    #define constraints and parameters
    constraints= [
        {'constraint': 'f1 odd'},
        {'constraint': 'f2 odd'},
        {'constraint': 'f2(0)=0'},
    ]
    nn_params = {
        'neurons': 100,
        'layers': 3,
        'lr': 1e-4,
        'epochs': 2000,
        'device': 'cpu',
        'constraints': constraints,
    }

    #train a NN model
    models, evals, coefs = pycc.train(df, eqs, method='NN', params=nn_params)


Return Values
 

Training methods return a tuple of three variables: ``(models, evals, obtained_params)``.

* ``models``: A dictionary containing the trained models. This variable that depends on the chosen model. For example, for the **NN** method, this will be a dictionary mapping function names (e.g., ``'f1'``, and ``'f2'``) to their corresponding PyTorch ``NNModel`` objects. It is useful for using as input argument for pycc.simulate().   
* ``evals``: A flat list of NumPy arrays ready for plotting, containing the identified functions. It contains the x and y valures of each identified function in the format ``[x_f1, y_f1, x_f2, y_f2, ...]``. All training methods return this variable using the same format.  
* ``obtained_params``: A dictionary containing the identified scalar parameters from the equations (e.g., ``a1``, ``a2``). It contains the the parameter name (string) and its final identified value (float). All training methods return this variable using the same format.

**Method-Specific Details**

Below are the details and available ``params`` for each training method.

**Neural Network (method=\'NN\')**
----------------------------------

This method uses a physics-informed neural network to learn the unknown functions as characteristic curves. It is flexible and powerful for complex systems. The ``params`` dictionary for this method can contain the following keys:

* ``'neurons':`` (*int, optional*)
    The number of neurons in each hidden layer of the neural networks. **Default: 100**.
* ``'layers':`` (*int, optional*)
    The number of hidden layers for each neural network. **Default: 3**.
* ``'lr':`` (*float, optional*)
    The learning rate for the Adam optimizer. **Default: 1e-3**.
* ``'epochs':`` (*int, optional*) 
    The maximum number of training iterations. **Default: 1000**.
* ``'error_threshold':`` (*float, optional*)
    The training will stop early if the data loss falls below this value. **Default: 1e-6**.
* ``'device':`` (*str, optional*)
    Specifies the computation device. Options are ``'automatic'``, ``'cpu'``, ``'gpu'``, ``'cuda'``, and intel ``'xpu'``. The ``'gpu'`` setting search if any ``'cuda'`` or ``'xpu'`` are available. The ``'xpu'`` setting is based on intel-extension-for-pytorch and supports the following hardware: Intel Arc A- and B-Series, Iris Xe Graphics, Intel Data Center GPU Max Series (see more details in the `Intel documentation for pytorch extension <https://intel.github.io/intel-extension-for-pytorch/>`_). 
    
     The ``'automatic'`` setting will prioritize ``'cuda'``, ``'xpu'``, and finally ``'cpu'``. **Default: \'automatic\'**. 
* ``'eq_weights':`` (*list[float], optional*)
    A list of weights to apply to the loss function of each equation. The length of the list must match the number of equations. If not provided, all equations are weighted equally. **Default: \'None\'**.
* ``'weight_loss_param':`` (*float, optional*)
    A regularization factor for an L2 penalty on the identified scalar parameters (``a_i``). A small value helps prevent these parameters from growing too large. **Default: 1e-3**.
* ``'n_eval':`` (*int, optional*)
    The number of points to evaluate for generating the final characteristic curves in the ``evals`` output, i.e. the number of point to evaluate the obtained ``'fi'`` functions. **Default: 200**.
* ``'constraints':`` (*list[dict], optional*)
    A list of dictionaries, where each dictionary defines a physical constraint to be imposed on the model. This is one of the most powerful features. **Default: \'None\'**.

    Each constraint dictionary can have the following keys:
    * ``'constraint'``: A string defining the constraint. Supported formats are:
        * **Point value**: ``'f1(0)=0'`` or ``'f2(1.5)=-0.8'``
        * **Symmetry**: ``'f1 odd'`` or ``'f2 even'``
    * ``'penalty'``: (*float, optional*) A weight to multiply the loss from this specific constraint. **Default: 1.0**.
    * ``'eval'``: (*str, optional*) For symmetry constraints only. Defines how the constraint is evaluated. Can be ``'data'`` (uses the provided data points) or ``'array'`` (creates a new linearly spaced array over the data range). **Default: \'array\'**.
    * ``'Nval_array'``: (*int, optional*) If ``eval=\'array'``, this sets the number of points in the evaluation array. **Default: 100**.

    .. note::

        Here is an example of a ``constraints`` list:

        .. code-block:: python

            constraints = [
                # Force f1 to pass through the origin with a high penalty
                {'constraint': 'f1(0)=0', 'penalty': 100.0},

                # Enforce that f1 is an even function
                {'constraint': 'f1 even'},

                # Enforce that f2 is an odd function and add more customization
                {'constraint': 'f2 odd', 'eval': 'array', 'Nval_array': 200},
            ]


**Example Usage:**

.. code-block:: python

    # Assume 'df' is a pandas DataFrame with your data
    # df = pd.DataFrame(...)

    eqs = [
        'x1_dot = x2',
        'x2_dot = F_ext - f1(x2) - f2(x1)'
    ]

    constraints = [
        {'constraint': 'f1(0)=0', 'penalty': 100.0},
        {'constraint': 'f2 odd'},
    ]
    
    nn_params = {
        'neurons': 100,
        'layers': 3,
        'lr': 1e-4,
        'epochs': 2000,
        'device': 'cpu',
        'constraints': constraints,
    }

    models, evals, coefs = pycc.train(df, eqs, method='NN', params=nn_params)


**Symbolic Regression (method=\'SymbR\')**
-----------------------------------------

This method uses Symbolic Regression, powered by the ``PySR`` library, to discover concise mathematical expressions for the unknown functions. It works by iteratively by fitting a symbolic model for each function ``f_i`` to be compatible with the list of \'equations\' for a given input database.  

The ``params`` dictionary for this method can contain the following keys:

* ``'pysr':`` (*dict, optional*)
    A dictionary of keyword arguments that are passed directly to the ``PySRRegressor``. This is the primary way to control the symbolic regression process. See the `PySR documentation <https://astroautomata.com/PySR/api/>`_ for all options.
    PyCC code define:
    
    * ``'niterations'``: (*int*) Number of iterations for the search. **Default: 100**.
    * ``'unary_operators'``: (*list[str]*) A list of unary operators to use (e.g., ``'cos'``, ``'exp'``, ``'tanh'``). **Default: [\'tanh\']**.
    * ``'binary_operators'``: (*list[str]*) A list of binary operators to use (e.g., ``'+'``, ``'*'``). **Default: `[\'+\', \'-\', \'*\']**.
    * ``'maxsize'``: (*int*) The maximum complexity of the expressions. **Default: \'12\'**.
    * ``'populations'``: (*int*) The number of populations to use in the evolutionary search. **Default: 10**.
* ``'max_iterations':`` (*int, optional*)
    The maximum number of outer-loop iterations for the alternating fitting process. **Default: 15**.
* ``'tol':`` (*float, optional*)
    The tolerance for the change in mean squared error (MSE) between outer-loop iterations. The process will stop early if the change is less than this value. **Default: 1e-10**.
* ``'N_fit_points':`` (*int, optional*)
    The number of data points to subsample for fitting the PySR model. Using a smaller number can significantly speed up the process. If ``'None'``, all data is used. **Default: 200**.
* ``'scaling':`` (*bool, optional*)
    If ``'True'``, the input variable for each function is scaled to the range `[-1, 1]` before fitting, which can improve stability. **Default: \'False\'**.
* ``'n_eval':`` (*int, optional*)
    The number of points to evaluate for generating the final characteristic curves in the ``evals`` output. **Default: 200**.

**Example Usage:**

.. code-block:: python

    # Assume 'df' is a pandas DataFrame with your data
    # df = pd.DataFrame(...)

    eqs = [
        'x1_dot = x2',
        'x2_dot = F_ext - f1(x2) - f2(x1)'
    ]

    symbr_params = {
        'max_iterations': 20,
        'pysr': {
            'niterations': 50,
            'binary_operators': ["+", "*", "-", "/"],
            'unary_operators': ["cos", "sin", "tanh"],
            'maxsize': 15,
        }
    }

    models, evals, coefs = pycc.train(df, eqs, method='SymbR', params=symbr_params)


**Polynomial (method=\'Poly\')**
--------------------------------

This method models each unknown function ``f_i`` as a polynomial of a specified order. It uses an iterative Gauss-Newton algorithm to solve the potentially non-linear system of equations for the polynomial coefficients and any scalar parameters ``a_i``. This method is very fast and serves as a great baseline.

The ``params`` dictionary for this method can contain the following keys:

* ``'N_order':`` (*int, optional*)
    The order (highest degree) of the polynomial to fit for each function. **Default: 10**.
* ``'n_iter':`` (*int, optional*)
    The maximum number of iterations for the Gauss-Newton solver. **Default: 1000**.
* ``'learning_rate':`` (*float, optional*)
    The step size for the parameter update in each Gauss-Newton iteration. **Default: 0.01**.
* ``'error_threshold':`` (*float, optional*)
    The training will stop early if the total loss falls below this value. **Default: 1e-10**.
* ``'scaling':`` (*bool, optional*)
    If ``True``, the input variable for each function is scaled to the range `[-1, 1]` before fitting, which improves numerical stability for high-order polynomials. **Default: \'True\'**. 
* ``'eq_weights':`` (*list[float], optional*)
    A list of weights to apply to the loss function of each equation. **Default: \'None\'**.
* ``'n_eval':`` (*int, optional*)
    The number of points to evaluate for generating the final characteristic curves in the ``evals`` output. **Default: 200**.
* ``'constraints':`` (*list[dict], optional*)
    A list of dictionaries to impose constraints on the polynomial forms, effectively setting certain coefficients to zero. **Default: []**.

    Each constraint dictionary can have the following keys:
    * ``'constraint'``: A string defining the constraint. Supported formats are:
        * **Point value**: ``'f1(0)=0'`` (This forces the constant term to be zero).
        * **Symmetry**: ``'f1 odd'`` (This forces all even-power coefficients to be zero) or ``'f1 even'`` (This forces all odd-power coefficients to be zero).

**Example Usage:**

.. code-block:: python

    # Assume 'df' is a pandas DataFrame with your data
    # df = pd.DataFrame(...)

    eqs = [
        'x_ddot = F_ext - a1*f1(x_dot) - f2(x)'
    ]

    constraints = [
        # f1 is a damping term, so it must be an odd function.
        {'constraint': 'f1 odd'},
        # f2 is a restoring force, so it must be odd and pass through the origin.
        {'constraint': 'f2 odd'},
        {'constraint': 'f2(0)=0'},
    ]
    
    poly_params = {
        'N_order': 15,
        'n_iter': 1500,
        'learning_rate': 0.05,
        'constraints': constraints,
    }

    models, evals, coefs = pycc.train(df, eqs, method='Poly', params=poly_params)


--------------------------------------------------------------------------------

pycc.simulate()
==========

.. autofunction:: pycc.simulate




