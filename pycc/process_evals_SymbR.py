import numpy as np
import sympy as sp
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def fit_symbolic_regression(x_vals, y_vals, niterations=200, populations=15):
    """
    Fit a symbolic regression model to approximate NN-learned function.
    Returns:
        expr  (sympy expression)
        f_num (callable numpy function that returns scalars for scalar inputs)
        model (trained PySRRegressor)
    """
    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        binary_operators=["+", "-", "*"],  # avoid /, pow for stability
        unary_operators=["cos", "sin", "exp", "log", "sqrt","tanh"],
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        verbosity=0,
    )
    x_vals = np.array(x_vals).reshape(-1, 1)
    y_vals = np.array(y_vals)

    model.fit(x_vals, y_vals)
    expr = model.get_best()["sympy_format"]

    # Create callable function with proper error handling
    try:
        # Method 1: Direct conversion with proper modules
        x_sym = sp.Symbol('x')
        expr_sympy = sp.sympify(expr)
        expr_simplified = sp.simplify(expr_sympy)
        
        # Use 'numpy' module mapping for trigonometric functions
        f_num_raw = sp.lambdify(x_sym, expr_simplified, modules=['numpy'])
        
        # Create wrapper to ensure scalar output for scalar input
        def f_num(x_input):
            result = f_num_raw(x_input)
            # Ensure scalar output for scalar input
            if np.isscalar(x_input) and hasattr(result, '__len__'):
                return float(result[0]) if len(result) > 0 else float(result)
            elif np.isscalar(x_input):
                return float(result)
            else:
                return np.asarray(result, dtype=float)
        
        # Test the function with a simple value to ensure it works
        test_val = 0.5
        test_result = f_num(test_val)
        if not np.isscalar(test_result):
            raise ValueError("Function doesn't return scalar for scalar input")
        
    except Exception as e:
        print(f"Warning: lambdify failed with error: {e}")
        print("Falling back to numerical approximation...")
        
        # Fallback: Create a wrapper function that handles the conversion
        def f_num(x_input):
            # Handle scalar input specially for ODE integration
            if np.isscalar(x_input):
                x_array = np.array([[float(x_input)]])
                result = model.predict(x_array)
                return float(result[0])
            else:
                # Handle array input
                x_array = np.asarray(x_input, dtype=float)
                if x_array.ndim == 1:
                    x_array = x_array.reshape(-1, 1)
                return model.predict(x_array).flatten()
    
    return expr, f_num, model

def process_evals_SymbR(evals, function_names=None, sr_params=None, plot=True):
    """
    Fit SR to each f_i from evals list [x1, y1, x2, y2, ...].
    Returns dict {f_name: {"expr": sympy_expr, "func": callable, "model": model}}
    """
    if sr_params is None:
        sr_params = {}

    num_functions = len(evals) // 2
    sr_results = {}

    for i in range(num_functions):
        x_vals, y_vals = evals[2 * i], evals[2 * i + 1]
        f_name = function_names[i] if function_names else f"f{i+1}"

        print(f"\nRunning SR for {f_name}...")

        expr, f_num, model = fit_symbolic_regression(
            x_vals, y_vals,
            niterations=sr_params.get("niterations", 200),
            populations=sr_params.get("populations", 15),
        )

        sr_results[f_name] = {
            "expr": expr,
            "func": f_num,
            "model": model,
        }

        if plot:
            plt.figure()
            plt.plot(x_vals, y_vals, label=f"{f_name} (NN)")
            plt.plot(x_vals, model.predict(np.array(x_vals).reshape(-1, 1)),
                     "--", label=f"{f_name} (SR)")
            plt.xlabel("x")
            plt.ylabel(f_name)
            plt.legend()
            plt.title(f"Symbolic regression fit for {f_name}")
            plt.show()

    return sr_results

# Alternative robust function creator for edge cases
def create_robust_function(expr_str, model):
    """
    Creates a robust callable function from symbolic expression with fallbacks.
    Ensures scalar output for scalar input (required for ODE integration).
    """
    try:
        # Try the standard approach first
        x_sym = sp.Symbol('x')
        expr_sympy = sp.sympify(expr_str)
        expr_simplified = sp.simplify(expr_sympy)
        func_raw = sp.lambdify(x_sym, expr_simplified, modules='numpy')
        
        # Create wrapper to ensure proper scalar handling
        def func(x_input):
            if np.isscalar(x_input):
                result = func_raw(x_input)
                return float(result) if np.isscalar(result) else float(result[0])
            else:
                return np.asarray(func_raw(x_input), dtype=float)
        
        # Test function
        test_result = func(1.0)
        if np.isfinite(test_result) and np.isscalar(test_result):
            return func
            
    except Exception as e:
        print(f"Symbolic function creation failed: {e}")
    
    # Fallback to model prediction
    def model_fallback(x):
        if np.isscalar(x):
            x_input = np.array([[float(x)]])
            result = model.predict(x_input)
            return float(result[0])
        else:
            x_input = np.asarray(x, dtype=float)
            if x_input.ndim == 1:
                x_input = x_input.reshape(-1, 1)
            return model.predict(x_input).flatten()
    
    return model_fallback

# Example usage for ODE integration
def create_ode_system(sr_results):
    """
    Create an ODE system function that works properly with solve_ivp.
    """
    f1 = sr_results["f1"]["func"]
    f2 = sr_results["f2"]["func"]
    
    def ode_rhs(t, y):
        x, x_dot = y
        F_ext = np.sin(t)  # example forcing
        
        # Ensure scalar operations
        f1_val = f1(x_dot)
        f2_val = f2(x)
        
        # Make sure we return scalars
        if not np.isscalar(f1_val):
            f1_val = float(f1_val)
        if not np.isscalar(f2_val):
            f2_val = float(f2_val)
            
        x_ddot = F_ext - f1_val - f2_val
        
        return [x_dot, x_ddot]
    
    return ode_rhs
