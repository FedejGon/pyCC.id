# process_sr.py (or whatever you've named it)

import re
import warnings
from typing import Dict, Any, List, Callable

import numpy as np
import sympy as sp
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import sympy as sp

def _simplify_expression(expr_str: str, precision: int = 4) -> str:
    """
    Uses sympy to expand and simplify a symbolic expression for readability.

    Example: "(x0 - 0.1)*(x0 + 0.2)" -> "x0**2 + 0.1*x0 - 0.02"
    """
    try:
        # Define the variable PySR uses
        x0 = sp.Symbol('x0')
        
        # Convert the string into a sympy expression
        expr = sp.sympify(expr_str, locals={'x0': x0})
        
        # Expand products (this is the key simplification step)
        expr = sp.expand(expr)
        
        # Round all numbers in the expression for cleanliness
        simplified_expr = expr.xreplace({
            n: sp.Float(round(n, precision)) for n in expr.atoms(sp.Number)
        })

        return str(simplified_expr)
    except Exception:
        # If anything goes wrong, just return the original string
        return expr_str
        
def _extract_pysr_expression(pm: PySRRegressor) -> str:
    """Precisely extracts the single best symbolic equation string from a PySR model."""
    if pm is None:
        return "0.0"
    try:
        if hasattr(pm, 'get_best') and pm.get_best() is not None:
            best_equation_series = pm.get_best()
            if 'equation' in best_equation_series:
                return str(best_equation_series['equation'])
        if hasattr(pm, 'equations_') and not pm.equations_.empty:
            best_row = pm.equations_.iloc[pm.equations_['score'].idxmin()]
            if 'equation' in best_row:
                return str(best_row['equation'])
        warnings.warn("Could not find a 'best' equation. Defaulting to '0.0'.")
        return "0.0"
    except Exception as e:
        warnings.warn(f"An error occurred while extracting the PySR equation: {e}")
        return "0.0"

def _create_callable_from_model(expr_str: str, model: PySRRegressor) -> Callable:
    """Creates a robust callable function from a sympy expression, with a fallback to model.predict."""
    try:
        x_sym = sp.Symbol('x0') # Use 'x0' for consistency with PySR
        expr_sympy = sp.sympify(expr_str)
        func_raw = sp.lambdify(x_sym, expr_sympy, modules='numpy')
        
        # Test the lambdified function
        func_raw(1.0) 
        
        # Return the raw function, it's generally robust for arrays and scalars
        return func_raw
    except Exception as e:
        warnings.warn(f"Sympy lambdify failed ('{e}'). Falling back to model.predict().")
        
        # Fallback to a wrapper around model.predict()
        def model_fallback(x):
            x_arr = np.asarray(x, dtype=float)
            is_scalar = x_arr.ndim == 0
            
            # Ensure input is 2D for PySR
            y = model.predict(x_arr.reshape(-1, 1))
            
            # Return a scalar if the input was a scalar
            return float(y[0]) if is_scalar else y.flatten()
            
        return model_fallback

def post_processing_SymbR(equations: List[str], params: Dict[str, Any]):
    """
    Fits Symbolic Regression models to evaluated data points.
    
    This function takes evaluated data (from a previous step) and uses PySR
    to find symbolic expressions for the unknown functions f1, f2, etc.

    Args:
        equations: The list of system equation strings. Used to find function names.
        params: A dictionary containing the necessary parameters:
            - 'evals' (List): The data, structured as [x1, y1, x2, y2, ...].
            - 'pysr' (Dict): A dictionary of parameters to pass to PySRRegressor.
            - 'plot' (bool, optional): If True, shows plots of the fits. Defaults to True.

    Returns:
        A dictionary structured like {'f1': {...}, 'f2': {...}}, ready to be used
        as the 'models' parameter in the simulation function.
    """
    # --- 1. Unpack parameters from the params dictionary ---
    evals = params.get('evals')
    if evals is None:
        raise ValueError("The 'params' dictionary must contain the 'evals' key.")
        
    pysr_params = params.get('pysr', {})
    plot_fits = params.get('plot', True)

    # --- 2. Automatically find function names from the equation strings ---
    all_eq_str = " ".join(equations)
    # Find all occurrences of patterns like 'f1(..)', 'f2(..)', etc.
    function_names = sorted(list(set(re.findall(r'f\d+', all_eq_str))))
    
    if len(evals) // 2 != len(function_names):
        raise ValueError(f"Found {len(function_names)} function names but data for {len(evals)//2} functions.")

    # --- 3. Loop through each function, run PySR, and store results ---
    models_sr = {}
    for i, f_name in enumerate(function_names):
        x_vals, y_vals = evals[2 * i], evals[2 * i + 1]
        
        print(f"\nRunning Symbolic Regression for {f_name}...")

        # Initialize PySR with the flexible dictionary of parameters
        model = PySRRegressor(**pysr_params)
        
        # Fit the model
        model.fit(np.asarray(x_vals).reshape(-1, 1), np.asarray(y_vals))
        
        # Get the best symbolic expression as a string
        expr_str = _extract_pysr_expression(model)
        
        # Create a robust callable function from the result
        callable_func = _create_callable_from_model(expr_str, model)
        
        # Store the results in the format expected by the simulator
        models_sr[f_name] = {
            "expr": expr_str,
            "func": callable_func,
            "pysr_model": model,  # Store the full model for later inspection
        }

        # --- 4. Plot the results if requested ---
        if plot_fits:
            # Simplify the expression for the plot legend
            simplified_expr_str = _simplify_expression(expr_str)
            plt.figure(figsize=(6, 6))
            plt.plot(x_vals, y_vals, 'o', markersize=4, label=f"{f_name} evals")
            plt.plot(x_vals, callable_func(x_vals), "-", linewidth=2, label=f"SR Fit: {simplified_expr_str}")
            plt.xlabel("x")
            plt.ylabel(f_name)
            plt.title(f"Symbolic Regression Fit for {f_name}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

    return models_sr
