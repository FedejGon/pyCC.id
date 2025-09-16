# train_polynomial.py
import numpy as np
import re
import sympy as sp

# --- helpers (same as in your NN file) ---
def parse_functions(equation_str):
    """
    Return list of (f_name, var_name) in appearance order, e.g. [('f1','x_dot'), ('f2','x')]
    """
    pattern = r'(f\d+)\(([a-zA-Z_]+)\)'
    funcs = re.findall(pattern, equation_str)
    unique_funcs = list(dict.fromkeys(funcs))
    return unique_funcs

def extract_parameters(equation_str):
    """
    Return sorted list of symbolic scalar parameters a1,a2,...
    """
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))


def train_polynomial(df, equation_str, params=None):
    """
    General polynomial identification returning:
      models: dict f_name -> {'coeffs', 'A0','A1','var'}
      evals: [x_f1, f1_vals, x_f2, f2_vals, ...]  (same format as NN)
      scalar_coefs: dict of learned a_i -> float

    params:
      - N_order (default 10)
      - scaling (default True)
    """
    if params is None:
        params = {}
    N_order = int(params.get('N_order', 10))
    scaling = bool(params.get('scaling', True))

    # parse functions and params
    func_list = parse_functions(equation_str)       # list of (f_name, var_name)
    param_names = extract_parameters(equation_str)  # list of 'a1','a2',...

    if len(func_list) == 0 and len(param_names) == 0:
        raise ValueError("No f_i or a_i found in the equation string.")

    # Build symbolic expr = lhs - rhs
    eq_str = equation_str.replace('^', '**')
    lhs_str, rhs_str = eq_str.split('=')
    lhs = sp.sympify(lhs_str)
    rhs = sp.sympify(rhs_str)
    expr = sp.expand(lhs - rhs)

    # Build known_expr = expr with all f_i -> 0 and a_i -> 0, evaluate on dataframe to form b = -known_vals
    known_expr = expr
    for f_name, var_name in func_list:
        known_expr = known_expr.subs(sp.Function(f_name)(sp.Symbol(var_name)), 0)
    for a_name in param_names:
        known_expr = known_expr.subs(sp.Symbol(a_name), 0)
    known_expr = sp.simplify(known_expr)

    known_syms = sorted(list(known_expr.free_symbols), key=lambda s: str(s))
    if len(known_syms) == 0:
        known_vals = float(sp.N(known_expr)) * np.ones(len(df))
    else:
        # lambdify and evaluate in the order of known_syms
        lamb = sp.lambdify(tuple(known_syms), known_expr, 'numpy')
        inputs = [df[str(s)].values for s in known_syms]
        known_vals = np.asarray(lamb(*inputs), dtype=float).reshape(-1)
    b = -known_vals   # A @ coeffs = b

    # Build design matrix: blocks for each f_i (polynomial basis) and for each a_i (constant)
    N = len(df)
    A_blocks = []
    scaling_params = {}

    # For each f_i determine its symbolic multiplier in expr and create scaled polynomial block
    for f_name, var_name in func_list:
        # create symbol Fi and substitute to read multiplier
        Fi = sp.Symbol(f_name + '_sym')
        subs = {}
        for fn, vn in func_list:
            subs[sp.Function(fn)(sp.Symbol(vn))] = 0
        subs[sp.Function(f_name)(sp.Symbol(var_name))] = Fi
        for a_name in param_names:
            subs[sp.Symbol(a_name)] = 0
        coeff_expr = sp.expand(expr.subs(subs)).coeff(Fi)
        coeff_val = float(coeff_expr)  # numeric multiplier (sign preserved)

        # prepare data and scaling
        x = df[var_name].values.astype(float)
        if scaling:
            A0 = float((np.max(x) + np.min(x)) / 2.0)
            A1 = float((np.max(x) - np.min(x)) / 2.0)
            if A1 == 0:
                A1 = 1.0
        else:
            A0, A1 = 0.0, 1.0
        scaling_params[var_name] = (A0, A1)

        Z = np.vstack([((x - A0) / A1) ** i for i in range(N_order + 1)]).T  # (N, N_order+1)
        A_blocks.append(coeff_val * Z)

    # scalar parameters a_i: find their multipliers and add columns
    a_info = []
    for a_name in param_names:
        Ai = sp.Symbol(a_name + '_sym')
        subs = {}
        for fn, vn in func_list:
            subs[sp.Function(fn)(sp.Symbol(vn))] = 0
        for other in param_names:
            if other == a_name:
                subs[sp.Symbol(other)] = Ai
            else:
                subs[sp.Symbol(other)] = 0
        coeff_expr = sp.expand(expr.subs(subs)).coeff(Ai)
        coeff_val = float(coeff_expr)
        A_blocks.append(coeff_val * np.ones((N, 1)))
        a_info.append((a_name, coeff_val))

    if len(A_blocks) == 0:
        raise ValueError("No unknown f_i or a_i found in equation.")

    A = np.hstack(A_blocks)   # (N, total_unknowns)

    # Solve least squares
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Unpack coefficients
    coefs_funcs = {}
    idx = 0
    for f_name, var_name in func_list:
        coefs_funcs[f_name] = coeffs[idx: idx + N_order + 1].astype(float)
        idx += N_order + 1
    scalar_coefs = {}
    for (a_name, _) in a_info:
        scalar_coefs[a_name] = float(coeffs[idx])
        idx += 1

    # Build models dict with scaling info
    models = {}
    for f_name, var_name in func_list:
        models[f_name] = {
            'coeffs': np.asarray(coefs_funcs[f_name], dtype=float),
            'A0': float(scaling_params[var_name][0]),
            'A1': float(scaling_params[var_name][1]),
            'var': var_name
        }

    # Build evals list analogous to NN: for each f_i produce x_plot, y_plot (200 pts)
    evals = []
    for f_name, var_name in func_list:
        model = models[f_name]
        x_data = df[var_name].values.astype(float)
        x_min, x_max = np.min(x_data), np.max(x_data)
        x_plot = np.linspace(x_min, x_max, 200)
        c = model['coeffs']
        A0, A1 = model['A0'], model['A1']
        if A1 == 0:
            z = x_plot - A0
        else:
            z = (x_plot - A0) / A1
        Z = np.vstack([z**i for i in range(len(c))]).T
        y_plot = Z @ c
        evals.extend([x_plot, y_plot])

    # Print learned scalar parameters (similar to NN prints)
    for name, val in scalar_coefs.items():
        print(f"Learned {name}: {val:.6e}")

    return models, evals, scalar_coefs

