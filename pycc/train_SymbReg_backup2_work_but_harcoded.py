import time
import numpy as np
import pandas as pd
import re
import sympy as sp
from pysr import PySRRegressor
from pysindy.optimizers import STLSQ
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def parse_functions(equation_str):
    """Parse equation to find all f_i(variable) patterns"""
    pattern = r'(f\d+)\(([a-zA-Z_]+)\)'
    funcs = re.findall(pattern, equation_str)
    # Return unique functions preserving order
    unique_funcs = list(dict.fromkeys(funcs))
    return unique_funcs


def extract_parameters(equation_str):
    """Extract scalar parameters like a1, a2, etc."""
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))


def sympy_expression(equation_str):
    """Convert string equation to sympy expression"""
    equation_str = equation_str.replace('^', '**')
    lhs_str, rhs_str = equation_str.split('=')
    lhs = sp.sympify(lhs_str)
    rhs = sp.sympify(rhs_str)
    eq = sp.Eq(lhs, rhs)
    return eq


def prepare_data_arrays(df, variables):
    """Prepare numpy arrays from DataFrame columns"""
    arrays = {}
    for v in variables:
        if v in df.columns:
            arrays[v] = df[v].values
        else:
            raise ValueError(f"Variable {v} not found in DataFrame columns: {list(df.columns)}")
    return arrays


class SymbolicModel:
    """Wrapper class for symbolic expressions to mimic NN model interface"""
    def __init__(self, expr, variable):
        self.expr = expr
        self.variable = variable
        self.sympy_expr = expr
        self.func = sp.lambdify(variable, expr, "numpy")
    
    def __call__(self, x):
        """Evaluate the symbolic expression"""
        try:
            result = self.func(x)
            # Handle complex numbers that might arise from operations
            if np.iscomplexobj(result):
                result = result.real
            return result
        except (TypeError, ValueError, RuntimeWarning):
            # Return zeros if evaluation fails
            return np.zeros_like(x)
    
    def eval(self):
        """For compatibility with PyTorch models"""
        pass


def evaluate_target_from_equation(equation_str, data_arrays, known_functions=None):
    """
    Evaluate the target variable from the equation for regression.
    This assumes the equation is in the form: target = known_terms - unknown_functions
    """
    if known_functions is None:
        known_functions = {}
    
    # Parse the equation
    eq = sympy_expression(equation_str)
    lhs, rhs = eq.lhs, eq.rhs
    
    # For now, we'll use a simple approach where we assume the target is
    # everything except the unknown functions
    # This is a simplified implementation - you might need to customize this
    # based on your specific equation structure
    
    # Find all variables in the equation
    all_vars = list(eq.free_symbols)
    var_values = {}
    
    for var in all_vars:
        var_name = str(var)
        if var_name in data_arrays:
            var_values[var_name] = data_arrays[var_name]
    
    # This is a placeholder - in practice, you'd need more sophisticated
    # equation manipulation to isolate the target
    return np.zeros(len(next(iter(data_arrays.values()))))


def train_individual_function_pysr(X_data, y_data, variable_name, pysr_params):
    """Train a single function using PySR"""
    
    # Set up PySR with the provided parameters
    model = PySRRegressor(
        niterations=pysr_params.get('niterations', 50),
        binary_operators=pysr_params.get('binary_operators', ["+", "-", "*", "/"]),
        unary_operators=pysr_params.get('unary_operators', ["sin", "cos", "abs", "tanh"]),
        populations=pysr_params.get('populations', 40),
        population_size=pysr_params.get('population_size', 50),
        maxsize=pysr_params.get('maxsize', 15),
        progress=pysr_params.get('progress', False),
        model_selection=pysr_params.get('model_selection', "best"),
        verbosity=pysr_params.get('verbosity', 0)
    )
    
    # Fit the model
    X_reshaped = X_data.reshape(-1, 1) if X_data.ndim == 1 else X_data
    model.fit(X_reshaped, y_data, variable_names=[variable_name])
    
    return model


def combine_functions_with_stlsq(candidate_models, data_arrays, target_data, threshold=0.1, alpha=0.01):
    """Combine candidate functions using sparse regression (STLSQ)"""
    
    # Build feature matrix
    n_samples = len(target_data)
    n_candidates = len(candidate_models)
    Theta = np.zeros((n_samples, n_candidates))
    
    for i, (model, var_name) in enumerate(candidate_models):
        try:
            if var_name in data_arrays:
                Theta[:, i] = model(data_arrays[var_name])
            else:
                print(f"Warning: Variable {var_name} not found in data")
                Theta[:, i] = 0
        except Exception as e:
            print(f"Warning: Could not evaluate candidate {i}: {e}")
            Theta[:, i] = 0
    
    # Use STLSQ to find sparse combination
    optimizer = STLSQ(threshold=threshold, alpha=alpha)
    target_reshaped = target_data.reshape(-1, 1) if target_data.ndim == 1 else target_data
    optimizer.fit(Theta, target_reshaped)
    
    return optimizer.coef_[0], Theta


def calculate_equation_loss(equation_str, data_arrays, models, scalar_coefs=None):
    """Calculate the loss for a given equation using current models"""
    try:
        # Parse equation into sympy
        eq = sympy_expression(equation_str)
        lhs, rhs = eq.lhs, eq.rhs
        
        # Evaluate both sides
        lhs_val = evaluate_symbolic_expr(lhs, data_arrays, models, scalar_coefs)
        rhs_val = evaluate_symbolic_expr(rhs, data_arrays, models, scalar_coefs)
        
        # Calculate MSE
        residual = lhs_val - rhs_val
        mse = np.mean(residual**2)
        return mse, residual
    except Exception as e:
        print(f"Warning: Could not calculate loss: {e}")
        return float('inf'), np.zeros(len(next(iter(data_arrays.values()))))


def evaluate_symbolic_expr(expr, data_arrays, models, scalar_coefs=None):
    """Evaluate a sympy expression using data and models"""
    if scalar_coefs is None:
        scalar_coefs = {}
    
    # Convert sympy expression to numpy evaluation
    if expr.is_Number:
        return float(expr) * np.ones(len(next(iter(data_arrays.values()))))
    elif expr.is_Symbol:
        var_name = str(expr)
        if var_name in data_arrays:
            return data_arrays[var_name]
        elif var_name in scalar_coefs:
            return scalar_coefs[var_name] * np.ones(len(next(iter(data_arrays.values()))))
        else:
            return np.zeros(len(next(iter(data_arrays.values()))))
    elif expr.is_Function:
        func_name = expr.func.__name__
        if func_name in models:
            arg = expr.args[0]
            arg_val = evaluate_symbolic_expr(arg, data_arrays, models, scalar_coefs)
            return models[func_name](arg_val)
        else:
            # Handle standard mathematical functions
            return np.zeros(len(next(iter(data_arrays.values()))))
    elif expr.is_Add:
        result = np.zeros(len(next(iter(data_arrays.values()))))
        for arg in expr.args:
            result += evaluate_symbolic_expr(arg, data_arrays, models, scalar_coefs)
        return result
    elif expr.is_Mul:
        result = np.ones(len(next(iter(data_arrays.values()))))
        for arg in expr.args:
            result *= evaluate_symbolic_expr(arg, data_arrays, models, scalar_coefs)
        return result
    else:
        return np.zeros(len(next(iter(data_arrays.values()))))


def train_SymbReg(df, equation_str, params=None):
    """
    Main training function for Symbolic Regression approach with iterative refinement
    
    Args:
        df: DataFrame with data
        equation_str: String or list of equations
        params: Dictionary with parameters:
            - 'pysr': Dict of PySR parameters (niterations, operators, etc.)
            - 'N_fit_points': Number of data points to use for fitting
            - 'stlsq_threshold': STLSQ sparsity threshold
            - 'stlsq_alpha': STLSQ regularization parameter  
            - 'max_iterations': Maximum number of refinement iterations
            - 'error_threshold': Convergence threshold for early stopping
            - 'verbose': Whether to print detailed progress information
    
    Returns:
        models: Dict of symbolic models {func_name: SymbolicModel}
        evals: List of evaluation arrays for plotting [x1, y1, x2, y2, ...]
        scalar_coefs: Dict of scalar coefficients {param_name: value}
        
    The function performs iterative symbolic regression:
    1. Initialize with zero functions
    2. For each iteration:
       - Generate SR candidates for each unknown function
       - Use STLSQ to combine candidates optimally
       - Calculate and print equation loss
       - Keep best solution found so far
    3. Stop when converged or max iterations reached
    """
    
    if params is None:
        params = {}
    
    # Extract parameters
    pysr_params = params.get('pysr', {})
    N_fit_points = params.get('N_fit_points', len(df))
    stlsq_threshold = params.get('stlsq_threshold', 0.1)
    stlsq_alpha = params.get('stlsq_alpha', 0.01)
    max_iterations = params.get('max_iterations', 3)
    error_threshold = params.get('error_threshold', 1e-6)
    verbose = params.get('verbose', True)
    
    # Handle single equation or multiple equations
    if isinstance(equation_str, (list, tuple)):
        equations = list(equation_str)
    else:
        equations = [equation_str]
    
    # For now, we'll focus on the first equation (can be extended)
    main_equation = equations[0]
    
    print(f"Training SR model for equation: {main_equation}")
    
    # Parse functions from the main equation
    func_list = parse_functions(main_equation)
    param_names = extract_parameters(main_equation)
    
    print(f"Found functions: {func_list}")
    print(f"Found parameters: {param_names}")
    
    # Prepare data
    variables = list(df.columns)
    data_arrays = prepare_data_arrays(df, variables)
    
    # Subsample data if needed
    if N_fit_points < len(df):
        indices = np.linspace(0, len(df)-1, N_fit_points, dtype=int)
        for var in data_arrays:
            data_arrays[var] = data_arrays[var][indices]
    
    # Calculate target variable based on equation structure
    # This is a simplified approach - you may need to customize this
    # For the example equation 'x_ddot + f1(x_dot) + f2(x) - F_ext = 0'
    # The target would be: F_ext - x_ddot = f1(x_dot) + f2(x)
    
    if 'F_ext' in data_arrays and 'x_ddot' in data_arrays:
        target_data = data_arrays['F_ext'] - data_arrays['x_ddot']
    else:
        # Fallback: use zeros (you'll need to implement proper target calculation)
        target_data = np.zeros(len(data_arrays[variables[0]]))
        print("Warning: Using zero target - implement proper target calculation for your equation")
    
    models = {}
    scalar_coefs = {}
    
    # Initialize models with zero functions
    for func_name, var_name in func_list:
        models[func_name] = SymbolicModel(0, sp.Symbol(var_name))
    
    # Initialize scalar parameters
    for param in param_names:
        scalar_coefs[param] = 1.0
    
    # Iterative refinement loop
    best_loss = float('inf')
    best_models = None
    best_scalar_coefs = None
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*50}")
        
        all_candidates = []
        current_residual = target_data.copy()
        
        # Calculate current loss
        current_loss, residuals = calculate_equation_loss(main_equation, data_arrays, models, scalar_coefs)
        if verbose:
            print(f"Current equation loss: {current_loss:.6e}")
        
        # Stage 1: Generate candidates for each function separately
        for func_name, var_name in func_list:
            if verbose:
                print(f"\n--- Finding candidates for {func_name}({var_name}) ---")
            
            if var_name not in data_arrays:
                print(f"Warning: Variable {var_name} not found in data")
                continue
            
            # For iterative refinement, use current residuals as target
            if iteration > 0:
                # Calculate what this function should predict given current other functions
                other_predictions = np.zeros_like(target_data)
                for other_func, other_var in func_list:
                    if other_func != func_name and other_func in models:
                        other_predictions += models[other_func](data_arrays[other_var])
                
                function_target = target_data - other_predictions
            else:
                function_target = target_data
            
            # Train PySR model for this function
            X_data = data_arrays[var_name]
            
            # Adjust PySR parameters for refinement iterations
            current_pysr_params = pysr_params.copy()
            if iteration > 0:
                # Reduce iterations for refinement to speed up
                current_pysr_params['niterations'] = max(20, pysr_params.get('niterations', 50) // 2)
            
            # Train individual function
            sr_model = train_individual_function_pysr(X_data, function_target, var_name, current_pysr_params)
            
            # Extract candidate expressions
            if hasattr(sr_model, 'equations_'):
                equations_df = sr_model.equations_
                if verbose:
                    print(f"Found {len(equations_df)} candidates for {func_name}")
                
                # Convert to our symbolic model format
                for i, row in equations_df.iterrows():
                    try:
                        # Try to get sympy_format, fallback to equation if not available
                        if 'sympy_format' in row:
                            expr = row['sympy_format']
                        elif 'equation' in row:
                            expr_str = row['equation']
                            # Convert PySR equation string to sympy expression
                            expr_str = expr_str.replace(var_name, 'x')  # Normalize variable name
                            expr = sp.sympify(expr_str)
                            # Substitute back the correct variable
                            expr = expr.subs('x', sp.Symbol(var_name))
                        else:
                            continue
                        
                        symbolic_model = SymbolicModel(expr, sp.Symbol(var_name))
                        all_candidates.append((symbolic_model, var_name, func_name))
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not process candidate {i} for {func_name}: {e}")
                        continue
            else:
                if verbose:
                    print(f"Warning: No equations found for {func_name}")
        
        # Stage 2: Combine candidates using sparse regression
        if verbose:
            print(f"\n--- Combining {len(all_candidates)} candidates ---")
        
        if len(all_candidates) > 0:
            candidate_models = [(model, var_name) for model, var_name, _ in all_candidates]
            
            # Use STLSQ to find best combination
            coefficients, feature_matrix = combine_functions_with_stlsq(
                candidate_models, data_arrays, target_data, stlsq_threshold, stlsq_alpha
            )
            
            # Identify non-zero coefficients
            non_zero_indices = np.where(np.abs(coefficients) > 1e-5)[0]
            
            if verbose:
                print(f"Selected {len(non_zero_indices)} functions out of {len(coefficients)} candidates")
            
            # Build new models
            new_models = {}
            final_expressions = {}
            
            for func_name, var_name in func_list:
                final_expressions[func_name] = 0
                
            # Combine selected functions
            for idx in non_zero_indices:
                coeff = coefficients[idx]
                model, var_name, func_name = all_candidates[idx]
                
                if func_name not in final_expressions:
                    final_expressions[func_name] = 0
                
                final_expressions[func_name] += coeff * model.sympy_expr
                
                if verbose:
                    print(f"Selected: {coeff:.4f} * {model.sympy_expr} for {func_name}")
            
            # Create new symbolic models
            for func_name, var_name in func_list:
                if func_name in final_expressions and final_expressions[func_name] != 0:
                    new_models[func_name] = SymbolicModel(final_expressions[func_name], sp.Symbol(var_name))
                else:
                    # Keep previous model if nothing new was selected
                    if func_name in models:
                        new_models[func_name] = models[func_name]
                    else:
                        new_models[func_name] = SymbolicModel(0, sp.Symbol(var_name))
            
            # Calculate loss with new models
            new_loss, _ = calculate_equation_loss(main_equation, data_arrays, new_models, scalar_coefs)
            
            # Update if improvement
            if new_loss < best_loss:
                best_loss = new_loss
                best_models = new_models.copy()
                best_scalar_coefs = scalar_coefs.copy()
                models = new_models
                
                if verbose:
                    print(f"✓ Improved loss: {new_loss:.6e}")
            else:
                if verbose:
                    print(f"✗ No improvement: {new_loss:.6e} vs {best_loss:.6e}")
        
        # Check convergence
        if best_loss < error_threshold:
            if verbose:
                print(f"Converged at iteration {iteration + 1} with loss {best_loss:.6e}")
            break
    
    # Use best models found
    if best_models is not None:
        models = best_models
        scalar_coefs = best_scalar_coefs
    else:
        # Fallback to zero models if nothing was found
        if verbose:
            print("No candidates found, using zero models")
        models = {}
        scalar_coefs = {}
        for func_name, var_name in func_list:
            models[func_name] = SymbolicModel(0, sp.Symbol(var_name))
    
    # Final loss calculation and reporting
    final_loss, final_residuals = calculate_equation_loss(main_equation, data_arrays, models, scalar_coefs)
    
    # Print final results
    if verbose:
        print(f"\n{'='*50}")
        print(f"FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Final equation loss: {final_loss:.6e}")
        print(f"Converged in {iteration + 1} iterations" if best_loss < error_threshold else f"Completed {max_iterations} iterations")
        print(f"\n--- Final Identified Functions ---")
        
        for func_name in models:
            expr_str = str(models[func_name].sympy_expr)
            if expr_str != "0":
                print(f"{func_name}: {expr_str}")
            else:
                print(f"{func_name}: 0 (zero function)")
        
        if scalar_coefs:
            print(f"\n--- Scalar Parameters ---")
            for param, value in scalar_coefs.items():
                print(f"{param}: {value:.6f}")
    
    # Prepare evaluation arrays for plotting
    evals = []
    for func_name, var_name in func_list:
        if var_name in data_arrays and func_name in models:
            x_vals = data_arrays[var_name]
            x_plot = np.linspace(np.min(x_vals), np.max(x_vals), 200)
            y_plot = models[func_name](x_plot)
            evals.extend([x_plot, y_plot])
        else:
            # Add placeholder arrays
            evals.extend([np.linspace(-1, 1, 200), np.zeros(200)])
    
    return models, evals, scalar_coefs


def plot_results(models, evals, func_list, data_arrays=None, true_functions=None):
    """Plot the identified functions"""
    
    n_funcs = len(func_list)
    if n_funcs == 0:
        return
        
    fig, axes = plt.subplots(1, n_funcs, figsize=(5*n_funcs, 5))
    if n_funcs == 1:
        axes = [axes]
    
    eval_idx = 0
    for i, (func_name, var_name) in enumerate(func_list):
        ax = axes[i]
        
        # Plot identified function
        if eval_idx < len(evals):
            x_plot = evals[eval_idx]
            y_plot = evals[eval_idx + 1]
            ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f"Identified {func_name}")
            eval_idx += 2
        
        # Plot true function if provided
        if true_functions and func_name in true_functions:
            x_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
            y_true = true_functions[func_name](x_plot)
            ax.plot(x_plot, y_true, 'r--', linewidth=2, label=f"True {func_name}")
        
        # Plot data points if available
        if data_arrays and var_name in data_arrays:
            x_data = data_arrays[var_name]
            if func_name in models:
                y_data = models[func_name](x_data)
                ax.plot(x_data, y_data, 'k.', markersize=2, alpha=0.5, label="Data points")
        
        ax.set_xlabel(f"{var_name}")
        ax.set_ylabel(f"{func_name}({var_name})")
        ax.set_title(f"Function {func_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage function
def example_usage():
    """Example of how to use the train_SymbReg function"""
    
    # Create sample data
    t = np.linspace(0, 10, 1000)
    x = np.sin(t) * np.exp(-0.1*t)
    x_dot = np.gradient(x, t)
    x_ddot = np.gradient(x_dot, t)
    F_ext = np.cos(2*t)
    
    df = pd.DataFrame({
        't': t,
        'x': x,
        'x_dot': x_dot, 
        'x_ddot': x_ddot,
        'F_ext': F_ext
    })
    
    # Define equation and parameters
    equation = 'x_ddot + f1(x_dot) + f2(x) - F_ext = 0'
    params_SymbReg = {
        'pysr': {
            'niterations': 30,
            'unary_operators': ['tanh', 'sin', 'cos'],
            'binary_operators': ['+', '-', '*'],
            'maxsize': 15,
            'verbosity': 0,
            'progress': True
        },
        'N_fit_points': 200,
        'stlsq_threshold': 0.1,
        'stlsq_alpha': 0.01,
        'max_iterations': 3,
        'error_threshold': 1e-6,
        'verbose': True
    }
    
    print("Starting Symbolic Regression training...")
    print(f"Equation: {equation}")
    print(f"Data shape: {df.shape}")
    
    # Train the model
    models, evals, scalar_coefs = train_SymbReg(df, equation, params_SymbReg)
    
    # Plot results
    func_list = parse_functions(equation)
    plot_results(models, evals, func_list)
    
    return models, evals, scalar_coefs


if __name__ == "__main__":
    # Run example
    print("="*60)
    print("SYMBOLIC REGRESSION TRAINING EXAMPLE")
    print("="*60)
    
    models, evals, scalar_coefs = example_usage()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models found: {list(models.keys())}")
    print(f"Scalar coefficients: {list(scalar_coefs.keys())}")
    
    # Example with custom loss tracking parameters
    print("\n" + "="*60)  
    print("CUSTOM PARAMETERS EXAMPLE")
    print("="*60)
    
    # More intensive search with detailed loss tracking
    custom_params = {
        'pysr': {
            'niterations': 30,
            'unary_operators': ['tanh', 'sin', 'cos', 'exp'],
            'binary_operators': ['+', '-', '*', '/'],
            'maxsize': 12,
            'verbosity': 0,  # Keep PySR quiet, we'll handle our own printing
            'progress': False
        },
        'N_fit_points': 300,
        'stlsq_threshold': 0.05,  # More stringent selection
        'stlsq_alpha': 0.005,
        'max_iterations': 15,      # More iterations
        'error_threshold': 1e-8,  # Stricter convergence
        'verbose': True           # Detailed progress printing
    }
    
    # You can call it like this for more detailed tracking:
    # models2, evals2, scalar_coefs2 = train_SymbReg(df, equation, custom_params)
