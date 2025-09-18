# train_polynomial_linear.py
import numpy as np
import re
import sympy as sp

# --- helpers (same as in your NN file) ---
def parse_functions(equation_str):
    """
    Return list of (f_name, var_name) in appearance order, e.g. [('f1','x_dot'), ('f2','x')]
    """
    #pattern = r'(f\d+)\(([a-zA-Z_]+)\)' #without_numbers
    pattern = r'(f\d+)\((\w+)\)' # with numbers, i.e. f(x1)
    funcs = re.findall(pattern, equation_str)
    unique_funcs = list(dict.fromkeys(funcs))
    return unique_funcs

def extract_parameters(equation_str):
    """
    Return sorted list of symbolic scalar parameters a1,a2,...
    """
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))


def apply_constraints_poly(constraints, f_name, coeffs):
    """
    Enforce simple constraints directly on polynomial coefficients.
    - "f1 odd": only odd powers allowed (zero out even terms)
    - "f1 even": only even powers allowed (zero out odd terms)
    - "f1(0)=0": force constant term to zero
    """
    if constraints is None:
        return coeffs

    coeffs = coeffs.copy()
    for cons in constraints:
        rule = cons.get("constraint", "")
        if not rule.startswith(f_name):
            continue

        if rule.endswith("odd"):
            coeffs[::2] = 0.0  # zero even-power coefficients
        elif rule.endswith("even"):
            coeffs[1::2] = 0.0  # zero odd-power coefficients
        elif rule.endswith("(0)=0"):
            coeffs[0] = 0.0  # zero constant term
    return coeffs


def train_polynomial_linear(df, equation_str, params=None):
    """
    General polynomial identification:
      - accepts equation_str as str OR list/tuple of strings (backward compatible)
      - params keys:
          - N_order (default 10)
          - scaling (default True)
          - constraints: list of dicts (applied exactly: odd/even/zero)
          - eq_weights: scalar or list of weights per equation (default all 1)
    Returns:
      models: dict f_name -> {'coeffs', 'A0','A1','var'}
      evals: [x_f1, f1_vals, x_f2, f2_vals, ...]
      scalar_coefs: dict of learned a_i -> float
    """
    if params is None:
        params = {}

    N_order = int(params.get('N_order', 10))
    scaling = bool(params.get('scaling', True))
    constraints = params.get('constraints', [])
    eq_weights = params.get('eq_weights', None)

    # Accept single or multiple equations
    if isinstance(equation_str, (list, tuple)):
        equations = list(equation_str)
    else:
        equations = [equation_str]

    # parse global function map and ensure consistency (f_name -> var_name)
    func_map = {}
    func_order = []  # list of (f_name, var_name) preserving first appearance
    for eq in equations:
        fl = parse_functions(eq)
        for f_name, var_name in fl:
            if f_name in func_map:
                if func_map[f_name] != var_name:
                    raise ValueError(f"Function {f_name} used with different arguments: "
                                     f"{func_map[f_name]} vs {var_name}")
            else:
                func_map[f_name] = var_name
                func_order.append((f_name, var_name))

    # collect scalar parameter names across all equations
    param_names_set = set()
    for eq in equations:
        for p in extract_parameters(eq):
            param_names_set.add(p)
    param_names = sorted(param_names_set)

    # prepare per-f_name scaling bases (Z matrices) using df and func_map
    Z_map = {}           # f_name -> Z matrix shape (N, N_order+1)
    scaling_params = {}  # var_name -> (A0, A1)
    N = len(df)
    for f_name, var_name in func_order:
        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' used in {f_name} not found in dataframe columns")
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
        Z_map[f_name] = Z

    # build eq_weights list matching number of equations
    if eq_weights is None:
        weights = [1.0] * len(equations)
    elif np.isscalar(eq_weights):
        weights = [float(eq_weights)] * len(equations)
    else:
        if len(eq_weights) != len(equations):
            raise ValueError(f"eq_weights length {len(eq_weights)} does not match number of equations {len(equations)}")
        weights = [float(w) for w in eq_weights]

    # For each equation, compute known_expr (with f_i and a_i zeroed), its b_j,
    # and the multipliers for each f_i and a_i, then build A_j block (N x total_unknowns).
    A_blocks_per_eq = []
    b_list = []

    total_unknowns = len(func_order) * (N_order + 1) + len(param_names)

    for eq_idx, (eq, w) in enumerate(zip(equations, weights)):
        eq_str = eq.replace('^', '**')
        lhs_str, rhs_str = eq_str.split('=')
        lhs = sp.sympify(lhs_str)
        rhs = sp.sympify(rhs_str)
        expr = sp.expand(lhs - rhs)

        # known_expr = expr with all f_i->0 and a_i->0
        known_expr = expr
        for fn, vn in func_order:
            known_expr = known_expr.subs(sp.Function(fn)(sp.Symbol(vn)), 0)
        for a_name in param_names:
            known_expr = known_expr.subs(sp.Symbol(a_name), 0)
        known_expr = sp.simplify(known_expr)

        known_syms = sorted(list(known_expr.free_symbols), key=lambda s: str(s))
        if len(known_syms) == 0:
            known_vals = float(sp.N(known_expr)) * np.ones(N)
        else:
            lamb = sp.lambdify(tuple(known_syms), known_expr, 'numpy')
            inputs = [df[str(s)].values for s in known_syms]
            known_vals = np.asarray(lamb(*inputs), dtype=float).reshape(-1)

        b_j = -known_vals  # shape (N,)
        # apply weight by scaling rows: multiply both A_j and b_j by sqrt(weight)
        sqrt_w = np.sqrt(w) if w > 0 else 0.0

        # Build A_j by concatenating columns for each f in func_order and each a in param_names
        row_blocks = []
        # multipliers for each f in this equation
        for f_name, var_name in func_order:
            # substitute other f->0 and a->0, replace this f by symbol Fi to extract multiplier
            Fi = sp.Symbol(f_name + '_sym')
            subs = {}
            for fn, vn in func_order:
                subs[sp.Function(fn)(sp.Symbol(vn))] = 0
            subs[sp.Function(f_name)(sp.Symbol(var_name))] = Fi
            for a_name in param_names:
                subs[sp.Symbol(a_name)] = 0
            coeff_expr = sp.expand(expr.subs(subs)).coeff(Fi)
            coeff_val = float(coeff_expr)  # multiplier (could be +/- or numeric)
            Z = Z_map[f_name]  # (N, N_order+1)
            block = coeff_val * Z  # (N, N_order+1)
            if sqrt_w != 1.0:
                block = block * sqrt_w
            row_blocks.append(block)

        # columns for a_i (each is coeff * 1)
        for a_name in param_names:
            Ai = sp.Symbol(a_name + '_sym')
            subs = {}
            for fn, vn in func_order:
                subs[sp.Function(fn)(sp.Symbol(vn))] = 0
            for other in param_names:
                subs[sp.Symbol(other)] = Ai if other == a_name else 0
            coeff_expr = sp.expand(expr.subs(subs)).coeff(Ai)
            coeff_val = float(coeff_expr)
            col = coeff_val * np.ones((N, 1))
            if sqrt_w != 1.0:
                col = col * sqrt_w
            row_blocks.append(col)

        A_j = np.hstack(row_blocks)  # shape (N, total_unknowns)
        # scale b_j by sqrt_w
        if sqrt_w != 1.0:
            b_j = b_j * sqrt_w

        A_blocks_per_eq.append(A_j)
        b_list.append(b_j)

    # Stack vertically across equations
    A_total = np.vstack(A_blocks_per_eq)  # (N * n_eq, total_unknowns)
    b_total = np.concatenate(b_list)      # (N * n_eq,)

    # Solve least squares
    coeffs, *_ = np.linalg.lstsq(A_total, b_total, rcond=None)

    # Unpack coefficients into per-function arrays and scalar params, then apply constraints
    coefs_funcs = {}
    idx = 0
    for f_name, var_name in func_order:
        raw = coeffs[idx: idx + N_order + 1].astype(float)
        constrained = apply_constraints_poly(constraints, f_name, raw)
        coefs_funcs[f_name] = constrained
        idx += N_order + 1

    scalar_coefs = {}
    for (a_name) in param_names:
        scalar_coefs[a_name] = float(coeffs[idx]) if idx < len(coeffs) else 0.0
        idx += 1

    # Build models dict with scaling info
    models = {}
    for f_name, var_name in func_order:
        A0, A1 = scaling_params[var_name]
        models[f_name] = {
            'coeffs': np.asarray(coefs_funcs[f_name], dtype=float),
            'A0': float(A0),
            'A1': float(A1),
            'var': var_name
        }

    # Build evals list analogous to NN: for each f_i produce x_plot, y_plot (200 pts)
    evals = []
    for f_name, var_name in func_order:
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
        Zp = np.vstack([z**i for i in range(len(c))]).T
        y_plot = Zp @ c
        evals.extend([x_plot, y_plot])

    # Print learned scalar parameters (similar to NN prints)
    for name, val in scalar_coefs.items():
        print(f"Learned {name}: {val:.6e}")

    return models, evals, scalar_coefs

