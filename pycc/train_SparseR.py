import numpy as np
import re
import sympy as sp
from collections import OrderedDict
from sklearn.linear_model import Lasso

# --- Internal helper functions for library generation ---
def _polynomials(order):
    if not isinstance(order, int) or order < 1:
        return []
    return [f'x**{i}' for i in range(1, order + 1)]

def _sines(max_freq):
    if not isinstance(max_freq, int) or max_freq < 1:
        return []
    return [f'sin({k}*x)' for k in range(1, max_freq + 1)]

def _cosines(max_freq):
    if not isinstance(max_freq, int) or max_freq < 1:
        return []
    return [f'cos({k}*x)' for k in range(1, max_freq + 1)]

def parse_functions(equation_str):
    """
    Parses an equation string to find all function calls like f1(x), f2(y).
    Returns a list of unique (function_name, variable_name) tuples.
    """
    pattern = r'(f\d+)\((\w+)\)'
    all_funcs = re.findall(pattern, equation_str)
    unique_funcs = list(OrderedDict.fromkeys(all_funcs))
    return unique_funcs

def extract_parameters(equation_str):
    """Parses for scalar parameters like a0, a1, etc."""
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))

def train_SparseR(df, equations, params=None):
    """
    Identifies unknown functions and parameters in an ODE using sparse regression (LASSO).
    The equation with unknown terms should be in the form: derivative = expression.
    """
    if params is None:
        params = {}

    # --- Hyperparameters ---
    alpha = float(params.get('alpha', 1e-2))
    n_eval = int(params.get('n_eval', 200))
    library_config = params.get('library', {})
    tol = float(params.get('tol', 1e-6))
    max_iter = int(params.get('max_iter', 10000))

    if not library_config:
        print("Warning: No function library specified in params. Using default polynomial library.")
        library_config['default'] = ['polynomials(5)']

    # --- 1. Find the target equation and parse all components ---
    target_eq = None
    all_funcs_map = OrderedDict()
    all_params_list = []
    
    for eq in equations:
        funcs = parse_functions(eq)
        if funcs and target_eq is None:
            target_eq = eq
            all_funcs_map = OrderedDict(funcs)
        elif funcs and target_eq is not None:
             print(f"Warning: Multiple equations with unknown functions found. Ignoring subsequent one: '{eq}'")
        all_params_list.extend(extract_parameters(eq))

    if target_eq is None:
        raise ValueError("No equation with unknown functions (e.g., f1(x)) found.")
    
    all_params_list = sorted(list(set(all_params_list)))
    print(f"Identifying unknown functions {list(all_funcs_map.keys())} in equation: {target_eq}")

    # --- 2. Symbolic Analysis and Target (Y) Definition ---
    lhs_str, rhs_str = target_eq.split('=', 1)
    df_syms = {col: sp.Symbol(col) for col in df.columns}
    a_syms = {p: sp.Symbol(p) for p in all_params_list}
    f_classes = {f_name: sp.Function(f_name) for f_name in all_funcs_map.keys()}
    
    full_expr = sp.sympify(lhs_str, locals=df_syms) - sp.sympify(rhs_str, locals={**df_syms, **a_syms, **f_classes})
    
    # --- 3. Advanced Error Checking for Non-Linearities ---
    # Check for terms with multiple unknown functions (e.g., f1*f2)
    for term in sp.Add.make_args(full_expr):
        present_funcs = {f.name for f in term.atoms(sp.Function) if f.name in all_funcs_map}
        if len(present_funcs) > 1:
            raise NotImplementedError(
                f"Equation contains term '{term}' with multiple unknown functions {present_funcs}. "
                "The current linear sparse regression method cannot solve for non-linear combinations "
                "like f1*f2 or f1/f2. Consider using a non-linear method like genetic programming."
            )
    
    # Check for functions inside other functions (e.g., exp(f1))
    for f_name, var_name in all_funcs_map.items():
        f_instance = f_classes[f_name](df_syms[var_name])
        for atom in full_expr.atoms(sp.Function):
            if atom != f_instance and f_instance in atom.args:
                raise NotImplementedError(
                    f"Equation contains nested function '{atom}'. The linear sparse regression solver "
                    "cannot handle expressions like exp(f1(x)) or sin(f1(x)), where an unknown function "
                    "is an argument to another function. This requires a non-linear identification method."
                )

    # --- 4. Separate Known and Unknown Terms ---
    known_terms_expr = sp.S(0)
    unknown_terms_expr = sp.S(0)
    for term in sp.Add.make_args(full_expr):
        has_unknown_func = any(f.name in all_funcs_map for f in term.atoms(sp.Function))
        if has_unknown_func:
            unknown_terms_expr += term
        else:
            known_terms_expr += term

    Y_expr = -known_terms_expr
    Y_lambda = sp.lambdify(list(df_syms.values()), Y_expr, 'numpy')
    Y = Y_lambda(*[df[col].values for col in df_syms.keys()])
    
    print(f"Regression target Y defined as: {Y_expr}")
    print(f"Library Theta will be built to model: {unknown_terms_expr}")

    # --- 5. Build the Library of Candidate Functions (Theta) ---
    library_funcs = []
    library_syms = []
    library_term_map = []
    
    _LIB_GENERATORS = {'polynomials': _polynomials, 'sines': _sines, 'cosines': _cosines}

    for f_name, var_name in all_funcs_map.items():
        f_instance = f_classes[f_name](df_syms[var_name])
        coeff_expr = sp.diff(unknown_terms_expr, f_instance)
        
        if any(f.name in all_funcs_map for f in coeff_expr.atoms(sp.Function)):
            raise NotImplementedError(f"Coefficient of {f_name} is '{coeff_expr}', which depends on other unknown functions. This structure is not supported.")

        coeff_lambda = sp.lambdify(list(df_syms.values()), coeff_expr, 'numpy')
        coeff_values = coeff_lambda(*[df[col].values for col in df_syms.keys()])
        if not isinstance(coeff_values, np.ndarray):
            coeff_values = np.full(len(df), float(coeff_values))

        print(f"Building library for '{f_name}({var_name})' with coefficient expression: {coeff_expr}")
        
        x_data = df[var_name].values
        basis_function_defs = library_config.get(f_name, library_config.get('default', []))
        
        expanded_basis_strings = []
        for item in basis_function_defs:
            match = re.match(r'(\w+)\((\d+)\)', item)
            if match and match.group(1) in _LIB_GENERATORS:
                func_name, arg = match.groups(); arg = int(arg)
                expanded_basis_strings.extend(_LIB_GENERATORS[func_name](arg))
            else:
                expanded_basis_strings.append(item)

        lib_var_sym = sp.Symbol('x')
        
        for basis_str in expanded_basis_strings:
            try:
                basis_expr = sp.sympify(basis_str, locals={'x': lib_var_sym, 'sin': sp.sin, 'cos': sp.cos, 'tanh': sp.tanh, 'exp': sp.exp})
                basis_lambda = sp.lambdify(lib_var_sym, basis_expr, 'numpy')
                basis_values = basis_lambda(x_data)
                library_funcs.append(coeff_values * basis_values)
                
                actual_var_sym = sp.Symbol(var_name)
                library_syms.append(basis_expr.subs({lib_var_sym: actual_var_sym}))
                library_term_map.append(f_name)
            except Exception as e:
                print(f"Warning: Could not parse basis function '{basis_str}'. Skipping. Error: {e}")

    if not library_funcs:
        raise ValueError("The feature library is empty. Check the 'library' definition in your parameters.")

    Theta = np.vstack(library_funcs).T
    print(f"Constructed library Theta with {Theta.shape[1]} candidate functions.")

    # --- 6. Fit the Sparse Regression Model ---
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter, tol=tol)
    model.fit(Theta, Y)
    
    Xi = model.coef_
    intercept = model.intercept_

    # --- 7. Decompose Results ---
    print("\n--- Sparse Regression Results ---")
    models = {f_name: {'expr': sp.S(0), 'var': var_name} for f_name, var_name in all_funcs_map.items()}
    final_scalars = {'intercept': intercept}

    for i, coef in enumerate(Xi):
        if np.abs(coef) > 1e-8:
            models[library_term_map[i]]['expr'] += coef * library_syms[i]

    for f_name, model_info in models.items():
        print(f"\nFinal expression for {f_name}({model_info['var']}):")
        sp.pprint(model_info['expr'], use_unicode=True)

    # --- 8. Generate Evaluation Data for Plotting ---
    evals = []
    for f_name, model_info in models.items():
        var_name = model_info['var']
        f_expr = model_info['expr']
        x_data_orig = df[var_name].values
        x_plot = np.linspace(x_data_orig.min(), x_data_orig.max(), n_eval)
        
        if f_expr.free_symbols:
            var_sym = sp.Symbol(var_name)
            f_lambda = sp.lambdify(var_sym, f_expr, 'numpy')
            y_plot = f_lambda(x_plot)
        else:
            y_plot = np.full_like(x_plot, float(f_expr))
        evals.extend([x_plot, y_plot])

    return models, evals, final_scalars


