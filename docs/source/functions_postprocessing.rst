
======================
pycc.post_processing()
======================
    
     
    
-----    
    
**Symbolic Regression (method=\'SymbR\')**
-----------------------------------------


This is a standalone utility function designed for a common post-processing workflow: converting the numerical characteristic curves (``evals``) from a \'NN\', \'Poly\' or other model into explicit symbolic expressions *before* running a simulation.


While ``pycc.simulate(method='SymbR')`` can do this on-the-fly, this function allows you to:
1.  Run the symbolic regression fit as a separate, explicit step.
2.  Inspect, analyze, and save the discovered symbolic functions.
3.  Receive plots of the fits to validate their quality.
4.  Get a ``models`` dictionary that can be fed into ``pycc.simulate(method='SymbR')``.
5.  Get a new, clean ``evals_sr`` list based on the *symbolic* fit, which can be used for fast interpolation-based simulation.
   

**Function Parameters**

This function is called as ``pycc.post_processing(equations, method='SymbR', params)``.

* ``equations``: (*list[str]*)
    The list of system equation strings (e.g., ``['x1_dot = x2', 'x2_dot = F_ext - f1(x2) - f2(x1)']``). This is used to automatically find the names of the functions to fit (e.g., ``'f1'``, ``'f2'``).

* ``params``: (*dict*)
    A dictionary containing the following keys:
    
    * ``'evals'``: (*list, required*)
        The `evals` flat list (e.g., ``[x_f1, y_f1, x_f2, y_f2, ...]``) returned from a ``pycc.train()`` run (e.g., from `method='NN'`).
    * ``'pysr'``: (*dict, required*)
        A dictionary of keyword arguments that are passed directly to the ``PySRRegressor``. This is the primary way to control the symbolic regression process. See the `PySR documentation <https://astroautomata.com/PySR/api/>`_ for all options.
        
        **Example:**
        
        .. code-block:: python

            pysr_settings = {
                'niterations': 500,
                'populations': 20,
                'binary_operators': ["+", "*", "-"],
                'unary_operators': ["tanh", "sin", "cos"],
                'maxsize': 20,
                'verbosity': 0
            }

    * ``'plot'``: (*bool, optional*)
        If ``True``, the function will display a Matplotlib plot for each function, showing the original ``evals`` data points and the resulting symbolic fit. **Default: True**.
    * ``'n_eval'``: (*int, optional*)
        The number of points to use for generating the new, smooth characteristic curves in the ``evals_sr`` output. **Default: 200**.        


**Return Value**

This function returns a tuple of two variables: ``(models_sr, evals_sr)``.

* ``models_sr``: (*dict*)
    A dictionary containing the symbolic regression results, formatted to be used directly by the simulation function. Its structure is:
    
    .. code-block:: python

        {
            'f1': {
                'expr': '0.5*x0 + 0.1*x0**3',  # The symbolic expression
                'func': <callable_function>,  # A python function of the expression
                'pysr_model': <PySRRegressor object> # The full trained model
            },
            'f2': { ... }
        }

    This dictionary can be passed directly to ``pycc.simulate`` as the ``'models'`` parameter.

* ``evals_sr``: (*list*)
    A new, flat list of NumPy arrays ``[x_f1_new, y_f1_new, x_f2_new, y_f2_new, ...]``. This list contains the *discovered symbolic functions* evaluated at ``n_eval`` points over their original domain. This is useful for plotting or for use with ``pycc.simulate(method='Interp')``.

**Workflow Example**

Here is the complete workflow:
1.  **Train** a model (like 'NN') to get numerical `evals`.
2.  **Post-process** the `evals` with `post_processing_SymbR` to get symbolic `models_sr`.
3.  **Simulate** using either the new `models_sr` (symbolic) or `evals_sr` (interpolation).

.. code-block:: python

    import pycc
    import numpy as np

    # --- Assume 'eqs', 'df_data', 't_span', 'y0', etc. are defined ---
    
    # --- 1. Train an NN model to get 'evals' ---
    
    nn_params = {'epochs': 2000, 'lr': 1e-3, ...}
    models_nn, evals_nn, coefs_nn = pycc.train(df_data, 
                                              eqs, 
                                              method='NN', 
                                              params=nn_params)

    # --- 2. Post-process 'evals_nn' to get symbolic models ---
    
    # Define settings for the new PySR fit
    pysr_settings = {
        'niterations': 500,
        'populations': 20,
        'binary_operators': ['+', '*', '-'],
        'unary_operators': ['tanh', 'sin','cos'],
        'maxsize': 20
    }
    
    # Define the 'params' dictionary for the post-processing function
    post_process_params = {
        'evals': evals_nn,
        'pysr': pysr_settings,
        'plot': True,  # This will show the plots of the fits
        'n_eval': 200,   # Generate 200 points for the new evals_sr
    }
    
    # Run the post-processing
    # This will print the fits and show plots
    models_sr, evals_sr = pycc.post_processing(eqs, method='SymbR', params=post_process_params)
    
    # `models_sr` now contains the symbolic functions
    # `evals_sr` now contains the smooth curves from those functions

    # --- 3. Simulate using the new symbolic models ---
    
    # Assume 'F_ext_func', 't_eval', 'y0' are defined
    
    # Option A: Simulate using the standard 'SymbR' method
    sim_params = {
        'models': models_sr,         # Use the new symbolic models
        'obtained_coefs': coefs_nn, # Use the scalars from the NN fit
        'local_funcs': {'F_ext': F_ext_func},
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
    }
    sol, derivs = pycc.simulate(eqs, method='SymbR', params=sim_params)    
    
    # Option B (Faster): Simulate by interpolating the new symbolic evals
    sim_params_interp = {
        'evals': evals_sr,           # Use the new symbolic evals
        'obtained_coefs': coefs_nn,  # ai coefficients obtained from training method 
        'local_funcs': {'F_ext': F_ext_func},
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
        'interp_method': 'pchip'
    }
    sol_interp, derivs_interp = pycc.simulate(eqs, 
                                              method='Interp', 
                                              params=sim_params_interp)

.. raw:: html

   <hr style="border: none; border-top: 4px dashed #bbb;">
   <hr style="border: none; border-top: 4px solid #bbb; width: 50%; margin: 20px auto;">
   <hr style="border: none; border-top: 4px dashed #bbb;">
    
