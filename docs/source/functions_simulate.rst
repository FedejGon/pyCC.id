


===============
pycc.simulate()
===============

This section provides a detailed reference for the ``pycc.simulate()`` function


--------
Overview
--------

The ``pycc.simulate()`` function is responsible for integrating forward system dynamics over time. Like ``pycc.train()``, it operates as a manager that dispatches the task to a specific simulation strategy based on the ``method`` argument.

**Key Use Cases:**

..    
     By comparing the simulated trajectory against your original data, you can validate the accuracy of the identified dynamics.
    This is the main function for simulating the system dynamics over time. It acts as a manager that calls a specific simulation method based on the ``method`` parameter.
    This function is typically used *after* ``pycc.train()`` to integrate the identified equations of motion using the trained models. It can also be used in a standalone \'Theoretical\' mode if all functions and parameters are already known.

1. Validation (Post-Training): After training a model (e.g., using ``pycc.train(eqs,method='NN')``), you can use ``pycc.simulate()`` to integrate the discovered equations. 
2. Theoretical Simulation: You can use ``method='Theoretical'`` to simulate a ground-truth dynamics as shown in the `Usage` tab. This method can be used to generate the training datasets or to integrate the ground-truth equations forward under other initial conditions or external forces to validate and compare with the predictions of the identified models.

Below, we show additional details about the inputs and outputs for the manager function:

.. autofunction:: pycc.simulate
   :noindex:

**Basic example usage**: The following example demonstrates how to integrate forward ground-truth equations, then, we define the Database for training and train a model using ``method='NN'``. After that, we integrate forward the obtained model with the same initial conditions that were used to generate the training dataset, and finally compare our predictions with the ground-truth forward integrations 


 define a second-order system structure with unknown functions (f1​,f2​), and train a model using the Neural Network method (below we will .


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
   ##############################################


   ##############################################
   # Forward integration with the obtained model:
   # Defining the simulation params using the obtained model
   params_NN_simul = {
       'models': models,
       'obtained_coefs': coefs,
       'local_funcs': {'F_ext': F_ext},
       't_span': t_span,
       'y0': y0,    # e.g., [1.0, 0.0] for [x1(0), x2(0)]
       't_eval': t_eval,
       'method': 'LSODA',  # This is the solver for scipy.solve_ivp
       'atol': 1e-8,
       'rtol': 1e-6,
       'check_nan': True
   }
   # Simulating the obtained model 
   sol, derivatives = pycc.simulate(eqs, method='NN', params=params_NN_simul)
   print("Integration success:", sol.success)
   # Extracting the results
   time_sim = sol.t
   x1_sim = sol.y[0]  # State variable x1(t)
   x2_sim = sol.y[1]  # State variable x2(t)
   # Derivatives can also be obtained if necessary
   x1_dot_sim = derivatives[0] # x1_dot(t)
   x2_dot_sim = derivatives[1] # x2_dot(t)
   ##############################################

   ##############################################  
   # Compare the prediction with the ground-truth
   plt.figure()
   plt.plot(time_sim, x1_sim, label="identified model")
   plt.plot(time_data, x_data, 'r--', label="ground-truth")
   plt.xlabel('t')
   plt.ylabel('x(t)')
   plt.legend()
   plt.show()

Return Values of ``pycc.simulate()``
 

All simulation methods return a tuple of two variables: ``(solution, derivatives)``.

* ``solution``: A ``scipy.integrate.OdeSolution`` object. This object contains the integration results, including the time points (``solution.t``) and the state variable values (``solution.y``). For a system with state ``[x1, x2]``, ``solution.y[0]`` will be the array for ``x1(t)`` and ``solution.y[1]`` will be the array for ``x2(t)``.
* ``derivatives``: A NumPy array containing the time derivatives of the state variables, evaluated at each time step in ``solution.t``. For a system with state ``[x1, x2]``, ``derivatives[0]`` will be the array for ``x1_dot(t)`` and ``derivatives[1]`` will be the array for ``x2_dot(t)``. This is useful for plotting or analysis without needing to re-calculate the derivatives.

---------------------------
**Method-Specific Details**
---------------------------

Below are the details and available ``params`` for each simulation method.

**Common Parameters for All Methods:**

All simulation methods accept the following keys in their ``params`` dictionary, which are passed to ``scipy.integrate.solve_ivp``:

* ``'t_span'``: (*tuple[float, float], required*)
    The interval of integration ``(t_start, t_end)``.
* ``'y0'``: (*list[float] or np.ndarray, required*)
    The initial state vector ``[x1(t0), x2(t0), ...]``. The order *must* match the order of the state variables implied by the ``equations`` list.
* ``'t_eval'``: (*np.ndarray, optional*)
    Time points where the solution is stored. If ``None``, the solver chooses the time points. **Default: None**.
* ``'method'``: (*str, optional*)
    The integration method passed to ``scipy.integrate.solve_ivp`` (e.g., ``'LSODA'``, ``'RK45'``, ``'DOP853'``). **Default: \'LSODA\'**.
* ``'atol'``: (*float, optional*)
    Absolute tolerance for the solver. **Default: 1e-8**.
* ``'rtol'``: (*float, optional*)
    Relative tolerance for the solver. **Default: 1e-6**.
* ``'check_nan'``: (*bool, optional*)
    If ``True``, raises a ``RuntimeError`` if the simulation produces ``NaN`` or ``Inf`` values. **Default: True**.

-----

**Neural Network (method=\'NN\')**
----------------------------------

This method simulates the system using the trained PyTorch Neural Network models obtained from ``pycc.train(method='NN')``.

The ``params`` dictionary for this method may contain:

* ``'models'``: (*dict, required*)
    The ``models`` dictionary returned by ``pycc.train()``. This contains the trained PyTorch ``NNModel`` objects for functions like ``'f1'``, ``'f2'``.
* ``'obtained_coefs'``: (*dict, optional*)
    The ``obtained_params`` dictionary returned by ``pycc.train()``. Contains identified scalar parameters (e.g., ``'a1'``). **Default: {}**.
* ``'local_funcs'``: (*dict, optional*)
    A dictionary for any *external* or *known* functions (like ``'F_ext'``) that were *not* identified by ``pycc.train()``. Functions in ``models`` will override any matching keys here. **Default: {}**.
* ... and all **Common Parameters** listed above (``'t_span'``, ``'y0'``, etc.).

**Example Usage:** See the main example above at pycc.simulate().

-----

**Symbolic Regression (method='SymbR')**
-----------------------------------------

This method simulates the system using symbolic expressions. It has two primary modes of operation:

1.  **Standard Simulation:** Uses the `PySRRegressor` models directly from the ``models`` dictionary returned by ``pycc.train(...,method='SymbR')``.
2.  **Post-processing Fit:** Uses the ``evals`` data (the list of ``[x, y]`` arrays) from *another* training method (like 'NN' or 'Poly'). It will perform a *new* symbolic regression fit on this data *first*, and then simulate using the resulting symbolic expressions.

The ``params`` dictionary for this method must contain either ``'models'`` or ``'evals'``:


1.  **Standard Simulation:** Uses the `PySRRegressor` models directly from the ``models`` dictionary returned by ``pycc.train(method='SymbR')``.
2.  **Post-processing Simulation:** Uses the ``evals`` data (the list of ``[x, y]`` arrays) from *another* training method (like 'NN' or 'Poly'). It will perform a *new* symbolic regression fit on the ``evals`` data *first*, and then simulate using the resulting symbolic expressions.

**Parameters for Both Modes**

The following parameters are used in both simulation modes:

* ``'obtained_coefs'``: (*dict, optional*)
    The ``obtained_params`` dictionary returned by ``pycc.train()``. Contains identified scalar parameters (e.g., ``'a1'``). **Default: {}**.
* ``'local_funcs'``: (*dict, optional*)
    A dictionary for any *external* or *known* functions (like ``'F_ext'``) that were *not* identified by ``pycc.train()``. **Default: {}**.
* ``'print_models'``: (*bool, optional*)
    If ``True``, prints the discovered or used symbolic expressions to the console before starting the simulation. **Default: True**.
* ... and all **Common Parameters** listed at the top (``'t_span'``, ``'y0'``, ``'t_eval'``, etc.).

---

**Mode 1: Standard Simulation (using \'models\')**

This is the standard mode when you have already trained a symbolic model. The simulation will use the expressions found during training.
 
* ``'models'``: (*dict, required*)
    The ``models`` object returned by ``pycc.train(method='SymbR')``. This contains the ``PySRRegressor`` objects or their callable equivalents.

**Example 1: Standard Simulation using a previous training with SymbR**

.. code-block:: python

    # Assume 'models_sr', 'coefs_sr' are outputs from
    # models_sr, evals_sr, coefs_sr = pycc.train(..., method='SymbR')
    #
    # Assume 'eqs', 't_span', 'y0', 't_eval', and a
    # callable 'F_ext_func' are defined.

    params_sim_sr = {
        'models': models_sr,
        'obtained_coefs': coefs_sr,
        'local_funcs': {'F_ext': F_ext_func},
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
    }

    sol, derivs = pycc.simulate(eqs, method='SymbR', params=params_sim_sr)

---

**Mode 2: Post-processing Simulation (using \'evals\')**

This mode allows you to take the numerical characteristic curves (``evals``) from *any* model (e.g., 'NN') and find new symbolic expressions for them *on the fly* before simulating.

* ``'evals'``: (*list, required for this mode*)
    The `evals` flat list (e.g., ``[x_f1, y_f1, x_f2, y_f2, ...]``) returned from *any* ``pycc.train()`` run. This data will be used to fit new PySR models.
* ``'function_names'``: (*list[str], optional*)
    Used *only* with ``evals``. A list of string names (e.S., ``['f1', 'f2']``) for the functions being fit from the ``evals`` data.
* ``'pysr'``: (*dict, optional*)
    Used *only* with ``evals``. A dictionary of keyword arguments passed to ``PySRRegressor`` to control the *new* fit. You can override any PySR setting, but the following defaults are used:
    
    * ``'niterations'``: (*int*) **Default: 200**.
    * ``'populations'``: (*int*) **Default: 15**.
    * ``'binary_operators'``: (*list[str]*) **Default: ``['+', '-', '*']``**.
    * ``'unary_operators'``: (*list[str]*) **Default: ``['cos', 'sin', 'exp', 'log', 'sqrt', 'tanh']``**.
    * ``'loss'``: (*str*) **Default: ``"loss(x, y) = (x - y)^2"``**.
    
    See the `PySR documentation <https://astroautomata.com/PySR/api/>`_ for all other options.

**Example 2: Post-processing simulation (using 'evals' from an NN fit)**

.. code-block:: python

    # Assume 'evals_nn', 'coefs_nn' are outputs from
    # method_nn, evals_nn, coefs_nn = pycc.train(..., method='NN')
    #
    # Assume 'eqs', 't_span', 'y0', 't_eval', and a
    # callable 'F_ext_func' are defined.

    params_sim_post_sr = {
        'evals': evals_nn,  # Use the NN's characteristic curves
        'function_names': ['f1', 'f2'], # Name the functions for fitting
        'obtained_coefs': coefs_nn,  # Use the scalars from the NN fit
        'local_funcs': {'F_ext': F_ext_func},
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
        # Control the new SR fit
        'pysr': {
            'niterations': 500,
            'populations': 20,
            'binary_operators': ["+", "*", "-"],
            'unary_operators': ["tanh", "sin"],
        }
    }

    # This will first fit f1 and f2 to the 'evals_nn' data using PySR,
    # then simulate the system with the new symbolic functions.
    sol, derivs = pycc.simulate(eqs, method='SymbR', params=params_sim_post_sr)


This method simulates the system using the symbolic expressions discovered by ``pycc.train(method='SymbR')``. The ``models`` dictionary from ``train`` (which contains ``PySRRegressor`` objects) is used to evaluate the functions.


-----

**Polynomial (method=\'Poly\')**
--------------------------------

This method simulates the system using the polynomial models (e.g., coefficients) identified by ``pycc.train(method='Poly')``.

The ``params`` dictionary for this method must contain:

* ``'models'``: (*dict, required*)
    The ``models`` dictionary returned by ``pycc.train(method='Poly')``.
* ``'obtained_coefs'``: (*dict, optional*)
    The ``obtained_params`` dictionary returned by ``pycc.train()``. **Default: {}**.
* ``'local_funcs'``: (*dict, optional*)
    A dictionary for any *external* or *known* functions (like ``'F_ext'``). **Default: {}**.
* ... and all **Common Parameters** listed above (``'t_span'``, ``'y0'``, etc.).

**Example Usage:**

.. code-block:: python

    # Assume 'models_poly', 'coefs_poly' are outputs from
    # models_poly, evals, coefs_poly = pycc.train(..., method='Poly')
    #
    # Assume 'eqs', 't_span', 'y0', 't_eval', and a
    # callable 'F_ext_func' are defined.

    params_sim_poly = {
        'models': models_poly,
        'obtained_coefs': coefs_poly,
        'local_funcs': {'F_ext': F_ext_func},
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
    }

    sol, derivs = pycc.simulate(eqs, method='Poly', params=params_sim_poly)

-----

**Interpolation (method='Interp')**
------------------------------------

This method provides a lightweight way to simulate the system dynamics using only the discrete characteristic curves (``evals``) returned from *any* ``pycc.train`` run (e.g., 'NN', 'Poly', 'SymbR').

It is often significantly faster than simulating with the analytical expressions from ``method='SymbR'`` because it bypasses the computational overhead of symbolic evaluation at each time step, while generally maintaining a high level of accuracy.

It does *not* require a ``models`` object. Instead, it creates a 1D numerical interpolant for each function ``f_i`` (e.g., ``f1(x)``, ``f2(x)``) from the provided ``[x, y]`` data points. The simulation then proceeds by calling these interpolants.



The ``params`` dictionary for this method must contain:

* ``'evals'``: (*list, required*)
    The `evals` flat list (e.g., ``[x_f1, y_f1, x_f2, y_f2, ...]``) returned from a ``pycc.train()`` run. This data is used to build the interpolants.
* ``'function_names'``: (*list[str], optional*)
    A list of string names (e.g., ``['f1', 'f2']``) for the functions in the ``evals`` list. If not provided, they are automatically named ``['f1', 'f2', ...]``.
* ``'interp_method'``: (*str, optional*)
    The interpolation method to use. Options are:
    
    * ``'pchip'``: (Default) PCHIP 1-D monotonic cubic interpolator. Preserves the shape of the data and avoids oscillations.
    * ``'linear'``: Linear interpolation.
    * ``'cubic'``: Cubic spline interpolation (as in `scipy.interpolate.interp1d`).
    * ``'spline'``: A smoothing spline (`scipy.interpolate.UnivariateSpline`).
* ``'smoothing'``: (*float, optional*)
    Used *only* when ``interp_method='spline'``. The smoothing factor passed to `UnivariateSpline`. **Default: 0.0**.
* ``'extrapolate'``: (*bool, optional*)
    If ``True``, the interpolator will extrapolate values beyond the range of the provided data. If ``False``, it will return the value at the data boundary. **Default: True**.
* ``'obtained_coefs'``: (*dict, optional*)
    The ``obtained_params`` dictionary returned by ``pycc.train()``. Contains identified scalar parameters (e.g., ``'a1'``). **Default: {}**.
* ``'local_funcs'``: (*dict, optional*)
    A dictionary for any *external* or *known* functions (like ``'F_ext'``). **Default: {}**.
* ``'print_models'``: (*bool, optional*)
    If ``True``, prints the interpolation method used for each function. **Default: True**.
* ... and all **Common Parameters** listed above (``'t_span'``, ``'y0'``, ``'t_eval'``, etc.).

**Example Usage (using 'evals' from an NN fit):**

.. code-block:: python

    # Assume 'evals_nn', 'coefs_nn' are outputs from
    # pycc.train(..., method='NN')
    #
    # Assume 'eqs', 't_span', 'y0', 't_eval', and a
    # callable 'F_ext_func' are defined.

    params_sim_interp = {
        'evals': evals_nn,          # Use the NN's characteristic curves
        'function_names': ['f1', 'f2'], # Name them for interpolation
        'obtained_coefs': coefs_nn,  # Use the scalars from the NN fit
        'local_funcs': {'F_ext': F_ext_func},
        
        'interp_method': 'pchip',   # Use shape-preserving cubic
        
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
    }

    # This will simulate the system by interpolating the 'evals_nn' data.
    sol, derivs = pycc.simulate(eqs, method='Interp', params=params_sim_interp)

-----

**Theoretical (method=\'Theoretical\')**
--------------------------------------

This method simulates the system of equations using user-provided, *known* Python functions for all unknown terms (e.g., ``f1``, ``f2``) and external forces (``F_ext``). It does *not* require any output from ``pycc.train()``. It is used for validating known models or simulating ideal systems.

The ``params`` dictionary for this method may contain:

* ``'local_funcs'``: (*dict, optional*)
    A dictionary mapping *all* function names in the equations (e.g., ``'f1'``, ``'f2'``, ``'F_ext'``) to their corresponding Python callable functions.
* ``'scalar_params'``: (*dict, optional*)
    A dictionary mapping scalar parameter names (e.g., ``'a1'``) to their float values.
    *Note: ``'obtained_coefs'`` can also be used as this key.* **Default: {}**.
* ... and all **Common Parameters** listed above (``'t_span'``, ``'y0'``, etc.).

**Example Usage:**

.. code-block:: python

    import pycc
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define the known theoretical functions
    def f1_theory(x_dot):
        return 0.1 * x_dot

    def f2_theory(x):
        return 0.5 * x + 0.2 * x**3

    def F_ext_theory(t):
        return 1.0 * np.cos(0.5 * t)

    eqs = [
        'x1_dot = x2',
        'x2_dot = F_ext - a1*f1(x2) - f2(x1)'
    ]

    sim_params_th = {
        'local_funcs': {
            'f1': f1_theory,
            'f2': f2_theory,
            'F_ext': F_ext_theory
        },
        'scalar_params': {'a1': 1.0},
        't_span': (0, 100),
        'y0': [1.0, 0.0],  # Initial state [x1(0), x2(0)]
        't_eval': np.linspace(0, 100, 1000)
    }

    sol, derivs = pycc.simulate(eqs, method='Theoretical', params=sim_params_th)
    
    # Extract results
    time_sim = sol.t
    x1_sim = sol.y[0]  # State variable x1(t)
    x2_sim = sol.y[1]  # State variable x2(t)
    # Derivatives are also returned
    x1_dot_sim = derivs[0] # x1_dot(t)
    x2_dot_sim = derivs[1] # x2_dot(t)
    
    # Plot solution
    plt.figure()
    plt.plot(time_sim, x1_sim, label="x(t) simulated")
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
.. raw:: html

   <hr style="border: none; border-top: 4px dashed #bbb;">
   <hr style="border: none; border-top: 4px solid #bbb; width: 50%; margin: 20px auto;">
   <hr style="border: none; border-top: 4px dashed #bbb;">

    

    
