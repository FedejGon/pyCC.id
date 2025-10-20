.. _api_reference
=============
API Reference
=============

This section provides a detailed reference for the core functions in the ``pycc`` library.

--------------------------------------------------------------------------------

.. automodule:: pycc
   :noindex:

pycc.train
==========

This is the main function for identifying system dynamics from data. It acts as a manager that calls a specific training method based on the ``method`` parameter.

.. autofunction:: train

.. raw:: html

   <br>

**Method-Specific Details**
---------------------------

Below are the details and required ``params`` for each available training method.

### Neural Network Method (`method='NN'`)

This method uses a physics-informed neural network to learn the unknown functions as characteristic curves. It is flexible and powerful for complex systems.

**Example Usage:**

.. code-block:: python

    import pycc
    import pandas as pd

    # Assume 'df' is a pandas DataFrame with your data
    # df = pd.DataFrame(...)

    eqs = [
        'x1_dot = x2',
        'x2_dot = F_ext - f1(x2) - f2(x1)'
    ]

    nn_params = {
        'neurons': 100,
        'layers': 3,
        'lr': 1e-4,
        'epochs': 2000,
        'device': 'cpu',
        'constraints': [{'constraint': 'f1 odd'}],
    }

    models, evals, coefs = pycc.train(df, eqs, method='NN', params=nn_params)


**Underlying Function:**

This method is implemented by the following function, which contains detailed documentation on all available hyperparameters in the ``params`` dictionary.

.. autofunction:: train_NN_hybrid

---

### Symbolic Regression Method (`method='SymbR'`)

This method uses Symbolic Regression (via PySR) to find a simple mathematical expression for the unknown functions. It's excellent for discovering interpretable models.

**Example Usage:**

.. code-block:: python

    # (Continuing from previous setup)

    symbr_params = {
        'pysr_params': {
            'niterations': 50,
            'binary_operators': ["+", "*", "-", "/"],
            'unary_operators': ["cos", "exp", "sin"],
        }
    }

    models, evals, coefs = pycc.train(df, eqs, method='SymbR', params=symbr_params)

**Underlying Function:**

.. autofunction:: train_SymbR

---

### Polynomial Method (`method='Poly'`)

This method fits a polynomial of a specified degree to the unknown functions. It is a fast and simple baseline method.

**Example Usage:**

.. code-block:: python

    # (Continuing from previous setup)

    poly_params = {
        'degree': 5,
    }

    models, evals, coefs = pycc.train(df, eqs, method='Poly', params=poly_params)

**Underlying Function:**

.. autofunction:: train_polynomial

---

### Linear Polynomial Method (`method='Poly_linear'`)

This is a specialized version of the polynomial method that performs a linear regression, which is even faster and useful for systems known to be linear.

**Underlying Function:**

.. autofunction:: train_polynomial_linear

--------------------------------------------------------------------------------

.. _simulate_api:

pycc.simulate
=============

This is the main function for simulating system dynamics. It can integrate known differential equations.

.. autofunction:: simulate

### Theoretical Method (`method='Theoretical'`)

This method uses a standard ODE solver (``scipy.integrate.solve_ivp``) to integrate a set of user-defined differential equations. This is useful for generating ground-truth data.

**Example Usage:**

.. code-block:: python

    import numpy as np

    def F_ext(t):
        return np.cos(t)

    def f1(x):
        return 0.1 * x

    def f2(x):
        return x + 0.2 * x**3

    equations = [
        'x1_dot = x2',
        'x2_dot = F_ext - f1(x2) - f2(x1)'
    ]

    sim_params = {
        't_span': (0, 20),
        'y0': [0.0, 0.0],
        't_eval': np.linspace(0, 20, 1000),
        'local_funcs': {'f1': f1, 'f2': f2, 'F_ext': F_ext}
    }

    solution, derivatives = pycc.simulate(equations, method="Theoretical", params=sim_params)
