============
pycc.train()
============
This section provides a detailed reference for the ``pycc.train()`` function



--------
Overview
--------
The ``pycc.train()`` function is the primary entry point for identifying system dynamics from data. It functions as a training manager (or dispatcher) that abstracts the complexity of specific algorithms. Instead of calling a specific algorithm directly, you specify the ``method`` parameter (e.g., 'NN' for Neural Networks), and the manager automatically routes the data and equations to the appropriate specialized training module.
 
This modular design allows users to seamlessly switch between different identification techniques (like Neural Networks, Polynomial Regression, or Symbolic Regression) without changing their data structures or equation definitions.
 
..   This is the main function for identifying system dynamics from data. It acts as a manager that calls a specific training method based on the ``method`` parameter. For example, when ``method='NN'``, we activate the training with the neural network method. Hence, can easily switch between different training methods. this is a manager of training methods that switches between different training methods. In the following we show this training manager do not kwow what .... 
    -------------------------------------------------
    Additional details for the manager function 
    -------------------------------------------------
    
Below, we show additional details about the inputs and outputs for the manager function: 

.. autofunction:: pycc.train 
   :noindex:
  

**Basic example usage:** 
The following example demonstrates how to set up a DataFrame, define a second-order system structure with unknown functions (f1​,f2​), and train a model using the Neural Network method (below we will .

.. code-block:: python


   import pycc
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   ##############################################
   # Generating the theoretical data (could also be manually generated  
   #                        with solve_ivp or experimentally measured)
   alpha=1.0;beta=0.2;delta=0.1;Omega=1.0;
   x0=0.0;v0=0.0; y0=[x0,v0] # initial conditions
   t_span=(0, 20); t_eval=np.linspace(*t_span, 1000)
   def F1_th(x_dot):
       return delta * x_dot + 0.5 * np.tanh(500*x_dot)
   def F2_th(x):
       return alpha * x + beta * x**3
   def F_ext(t):
       return np.cos(Omega * t)
   eqs_th = ['x1_dot = x2',
             'x2_dot = F_ext - f1(x2) - f2(x1)']
   params_th = {
       't_span': t_span,
       'y0': y0,  
       't_eval': t_eval,
       'method': 'LSODA',
       'local_funcs': {'f1': lambda t: F1_th(t),'f2': lambda t: F2_th(t),'F_ext': lambda t: F_ext(t)}
   }
   sol,derivatives = pycc.simulate(eqs_th,method="Theoretical", params=params_th)
   time_data    = sol.t
   x1_data      = sol.y[0]
   x2_data      = sol.y[1]
   x1_dot_data  = derivatives[0]
   x2_dot_data  = derivatives[1]
   F_ext_val    = F_ext(time_data)
   ##############################################

   ##############################################
   # Defining the database for training and the proposed equations 
   # To this aim, we need to define a pandas DataFrame using the variables that will be used for training
   df = pd.DataFrame({
       'x1':x1_data,
       'x2':x2_data,
       'x1_dot':x1_dot_data,
       'x2_dot':x2_dot_data,
       'F_ext': F_ext_val
   })
   # Now, we define the system structure we want to discover
   eqs = [
       'x1_dot = x2',
       'x2_dot = F_ext - f1(x2) - f2(x1)'
   ]
   ##############################################

   ##############################################
   # Setting constraints and parameters, in this case we will use NN-CC method
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
   ##############################################

   # Training the model
   models, evals, coefs = pycc.train(df, eqs, method='NN', params=nn_params)


   # Plotting the obtained functions and parameters
   # Define your custom inputs here (add more if you have more than 2 variables)
   true_funcs = [F1, F2] # names of the theoretical equations (if we have them)
   #true_funcs = [] # if we don't know them we can define an empty list
   x_labels = ['$x_2$', '$x_1$'] # corresponding variables
   # Calculate number of plots
   n_plots = len(evals) // 2
   fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6)) 
   # Ensure axes is a list even if there's only 1 plot
   if n_plots == 1:
       axes = [axes]
   # Plotting loop
   for i in range(n_plots):
       ax = axes[i]  
       # Extract x and y for this specific characteristic curve
       x_cc = evals[2 * i]
       f_cc = evals[2 * i + 1]
       # Plot learned functions
       ax.plot(x_cc, f_cc, label=f'$f_{i+1}$ learned')
       # Plot theory (assuming the function exists and is callable)
       if i < len(true_funcs) and true_funcs[i] is not None:
           ax.plot(x_cc, true_funcs[i](x_cc), '--', label=f'$f_{i+1}$ theory')
       # Apply custom labels or fallback to generic ones
       x_label = x_labels[i] if i < len(x_labels) else f'$x_{i+1}$'       
       ax.set_xlabel(x_label)
       ax.set_ylabel(f'$f_{i+1}({x_label.strip("$")})$')
       ax.legend()
   plt.tight_layout()
   plt.show()
   
   # Print learned parameters if there are any
   if 'obtained_coefs' in globals() and obtained_coefs:
       print("\nLearned scalar parameters:")
       for name, val in obtained_coefs.items():
           print(f"{name} = {val.item():.4f}")
   

Return Values ``pycc.train()``
 

All training methods return a tuple of three variables: ``(models, evals, obtained_params)``.

* ``models``: A dictionary containing the trained models. This variable that depends on the chosen model. For example, for the **NN** method, this is a dictionary mapping function names (e.g., ``'f1'``, and ``'f2'``) to their corresponding PyTorch ``NNModel`` objects. This models variable is useful for using as an input argument for pycc.simulate().   
* ``evals``: A flat list of NumPy arrays ready for plotting, containing the identified functions. It contains the x and y values of each identified function in the format ``[x_f1, y_f1, x_f2, y_f2, ...]``. All training methods return this list using the same format.  
* ``obtained_params``: A dictionary containing the identified scalar parameters from the equations (e.g., ``a1``, ``a2``). It contains the the parameter name (string) and its final identified value (float). All training methods return this variable using the same format. 

---------------------------
**Method-Specific Details**
---------------------------

While the specific parameterization varies by method, it is all encapsulated within the ``params`` dictionary. The sections below detail the specific arguments required inside ``params`` for each approach.

-----

**Neural Networks (method=\'NN\')**
----------------------------------

This method uses a physics-informed neural network to learn the unknown functions as characteristic curves. It is flexible and powerful for complex systems. The ``params`` dictionary for this method can contain the following keys:

* ``'neurons':`` (*int, optional*) Default: 100
    The number of neurons in each hidden layer of the neural networks.
* ``'layers':`` (*int, optional*)  Default: 3
    The number of hidden layers for each neural network.
* ``'activation':`` (*list[str], optional*) Default: ``'relu'``
    The activation functions to use. You can use any function available in the ``torch.nn.functional``  library  <https://docs.pytorch.org/docs/stable/nn.functional.html>. We recommend using ``'leaky_relu'`` , ``'relu'``, ``'tanh'``, or ``'rrelu'``. These were found to yield robust results for both chaotic and discontinuous systems in DOI:10.1007/s11071-025-11744-6 and 10.48550/arXiv.2601.21720.
* ``'lr':`` (*float, optional*) Default: 1e-4
    The learning rate for the Adam optimizer for the neural networks.
* ``'scalar_lr':`` (*float, optional*) Default: same value as ``'lr'``
    Specifies a separate learning rate for the Adam optimizer to use *only* for the scalar parameters. The code creates two distinct optimizers: one for the neural network models (using ``'lr'``) and one for the scalars (using ``'scalar_lr'``). This allows the user to train the scalar coefficients at a different rate than the ``'fi'`` functions. 
* ``'epochs':`` (*int, optional*) Default: 1000
    The maximum number of training iterations. 
* ``'error_threshold':`` (*float, optional*) Default: 1e-6
    The training will stop early if the data loss falls below this value. 
* ``'device':`` (*str, optional*) Default: ``'automatic'``
    Specifies the computation device. Options are ``'automatic'``, ``'cpu'``, ``'gpu'``, ``'cuda'``, and intel ``'xpu'``. The ``'gpu'`` setting search if any ``'cuda'`` or ``'xpu'`` are available. The ``'xpu'`` setting is based on intel-extension-for-pytorch and supports the following hardware: Intel Arc A- and B-Series, Iris Xe Graphics, Intel Data Center GPU Max Series (see more details in the `Intel documentation for pytorch extension <https://intel.github.io/intel-extension-for-pytorch/>`_). 
    
     The ``'automatic'`` setting will prioritize ``'cuda'``, ``'xpu'``, and finally ``'cpu'``. 
* ``'eq_weights':`` (*list[float], optional*) Default: ``'None'``
    A list of weights to apply to the loss function of each equation. The length of the list must match the number of equations. If not provided, all equations are weighted equally.
* ``'weight_loss_param':`` (*float, optional*) Default: 1e-3
    A weight factor for the identified scalar parameters (``ai``). A small value helps prevent these parameters from growing too large during training, and a big value increase the importance of ``ai`` parameters over ``fi`` functions.
* ``'initial_params':`` (*dict, optional*) Default: 1.0 for all parameters
    An array containing the scalar parameters (e.g., ``'a1'``, ``'a2'``,...) and their desired initial floating-point values. Example: \'initial_params\': {\'a1\': 1.5, \'a4\': 2.0}. 
* ``'n_eval':`` (*int, optional*) Default: 200
    The number of points to evaluate for generating the final characteristic curves in the ``evals`` output, i.e. the number of point to evaluate the obtained ``'fi'`` functions. 
* ``'constraints':`` (*list[dict], optional*) Default: ``'None'``
    A list of dictionaries, where each dictionary defines a physical constraint to be imposed on the model. This is one of the most powerful features. 

    Each constraint dictionary can have the following keys:
    * ``'constraint'``: A string defining the constraint. Supported formats are:
        * **Point value**: ``'f1(0)=0'`` or ``'f2(1.5)=-0.8'``
        * **Symmetry**: ``'f1 odd'`` or ``'f2 even'``
    * ``'penalty'``: (*float, optional*) Default: 1.0.  A weight to multiply the loss from this specific constraint.
    * ``'eval'``: (*str, optional*) Default: ``'array'``. For symmetry constraints only. Defines how the constraint is evaluated. Can be ``'data'`` (uses the provided data points) or ``'array'`` (creates a new linearly spaced array over the data range).
    * ``'Nval_array'``: (*int, optional*) Default: 100. If ``eval='array'``, this sets the number of points in the evaluation array.

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

-----

**Symbolic Regression (method=\'SymbR\')**
-----------------------------------------

This method uses Symbolic Regression, powered by the ``PySR`` library, to discover mathematical expressions for the unknown functions through an outer loop. It works by iteratively by fitting a symbolic model for each function :math:`f_i` to be compatible with the list of \'equations\'.  

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





-----

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




.. raw:: html

   <hr style="border: none; border-top: 4px dashed #bbb;">
   <hr style="border: none; border-top: 4px solid #bbb; width: 50%; margin: 20px auto;">
   <hr style="border: none; border-top: 4px dashed #bbb;">



