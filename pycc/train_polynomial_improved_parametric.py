import numpy as np
import re
import sympy as sp
from collections import OrderedDict

def parse_functions(equation_str):
    pattern = r'(f\d+)\((\w+)\)'
    all_funcs = re.findall(pattern, equation_str)
    unique_funcs = list(OrderedDict.fromkeys(all_funcs))
    return unique_funcs

def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))

def build_constraint_mask(constraints, f_name, n_coeffs):
    mask = np.ones(n_coeffs, dtype=bool)
    if not constraints:
        return mask
    for cons in constraints:
        rule = cons.get("constraint", "").strip().replace('\t', ' ')
        if not rule or not rule.startswith(f_name):
            continue
        if re.search(r'\bodd\b', rule):
            mask[::2] = False   # zero even powers (including constant) for odd functions
        elif re.search(r'\beven\b', rule):
            mask[1::2] = False  # zero odd powers for even functions
        elif re.search(r'\(0\)\s*=\s*0', rule):
            mask[0] = False
    return mask

def train_polynomial(df, equation, params=None):
    if params is None:
        params = {}

    N_order = int(params.get('N_order', 10))
    scaling = bool(params.get('scaling', True))
    constraints = params.get('constraints', [])
    eq_weights = params.get('eq_weights', None)
    n_iter = int(params.get('n_iter', 1000))
    learning_rate = float(params.get('learning_rate', 0.01))
    error_threshold = float(params.get('error_threshold', 1e-10))

    equations = list(equation) if isinstance(equation, (list, tuple)) else [equation]
    N_data = len(df)
    n_coeffs_per_func = N_order + 1

    # --- Parse functions and scalar parameters ---
    func_map = OrderedDict()
    param_names = set()
    for eq in equations:
        for f_name, var_name in parse_functions(eq):
            if f_name in func_map and func_map[f_name] != var_name:
                raise ValueError(f"Function {f_name} used with different arguments.")
            func_map[f_name] = var_name
        param_names.update(extract_parameters(eq))

    func_order = list(func_map.items())  # list of (f_name, var_name)
    param_names = sorted(list(param_names))
    n_params = len(param_names)

    # --- Data preparation & Z_map per function ---
    Z_map = {}
    scaling_params = {}
    for f_name, var_name in func_order:
        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' used in {f_name} not found in dataframe columns")
        x = df[var_name].values.astype(float)
        if scaling:
            A0 = float((np.max(x) + np.min(x)) / 2.0)
            A1 = float((np.max(x) - np.min(x)) / 2.0)
            if A1 == 0.0:
                A1 = 1.0
        else:
            A0, A1 = 0.0, 1.0
        scaling_params[var_name] = (A0, A1)
        z = (x - A0) / A1
        Z_map[f_name] = np.vstack([z**i for i in range(n_coeffs_per_func)]).T  # (N_data, n_coeffs_per_func)

    # --- Constraint masks and total unknown count ---
    coeffs_mask = {}
    n_active_coeffs_total = 0
    for f_name, _ in func_order:
        mask = build_constraint_mask(constraints, f_name, n_coeffs_per_func)
        coeffs_mask[f_name] = mask
        n_active_coeffs_total += int(np.sum(mask))

    total_unknowns = int(n_active_coeffs_total + n_params)

    # --- Symbolic preparation ---
    sym_exprs = []
    for eq_str in equations:
        lhs_str, rhs_str = eq_str.split('=', 1)
        expr = sp.sympify(lhs_str) - sp.sympify(rhs_str)
        sym_exprs.append(expr)

    f_syms = {f_name: sp.Symbol(f_name) for f_name, _ in func_order}
    a_syms = {a_name: sp.Symbol(a_name) for a_name in param_names}
    # Map function calls like f1(x) -> f1 (symbol)
    func_call_to_sym_map = {sp.Function(f_name)(sp.Symbol(var_name)): f_syms[f_name] for f_name, var_name in func_order}

    lambdas = {}
    for i, expr in enumerate(sym_exprs):
        expr_simple = expr.subs(func_call_to_sym_map)
        all_needed_syms = sorted(expr_simple.free_symbols, key=lambda s: s.name)
        # store argument symbols order
        lambdas[f'vars_{i}'] = all_needed_syms
        lambdas[f'R_{i}'] = sp.lambdify(all_needed_syms, expr_simple, 'numpy')
        for f_name, _ in func_order:
            lambdas[f'J_{i}_{f_name}'] = sp.lambdify(all_needed_syms, expr_simple.diff(f_syms[f_name]), 'numpy')
        for a_name in param_names:
            lambdas[f'J_{i}_{a_name}'] = sp.lambdify(all_needed_syms, expr_simple.diff(a_syms[a_name]), 'numpy')

    # --- Normalize penalties defaults ---
    for cons in constraints:
        cons.setdefault("penalty", 1.0)
        cons.setdefault("eval", "array")
        if cons["eval"] == "array":
            cons.setdefault("Nval_array", 50)

    # --- Iterative fitting (Gauss-Newton) ---
    rng = np.random.RandomState(0)
    p = rng.randn(total_unknowns) * 1e-2

    print("Starting non-linear polynomial fitting with constraints...")
    for iter_num in range(n_iter):
        # Unpack p into full coefficient vectors AND build index mapping for each f_name
        coeffs_map = {}
        coeffs_idx_map = {}   # f_name -> (start_idx_in_p, end_idx_in_p)
        param_idx_map = {}    # a_name -> column index in p
        start_idx = 0
        for f_name, _ in func_order:
            mask = coeffs_mask[f_name]
            n_active = int(np.sum(mask))
            end_idx = start_idx + n_active
            full_coeffs = np.zeros(n_coeffs_per_func, dtype=float)
            if n_active > 0:
                full_coeffs[mask] = p[start_idx:end_idx]
            coeffs_map[f_name] = full_coeffs
            coeffs_idx_map[f_name] = (start_idx, end_idx)
            start_idx = end_idx
        # scalar params appear after all function active coeffs
        # Ensure we slice exactly n_params elements
        scalar_slice = p[start_idx:start_idx + n_params] if n_params > 0 else np.array([])
        scalar_vals = {name: float(scalar_slice[i]) for i, name in enumerate(param_names)} if n_params > 0 else {}
        # build param index map
        for i, name in enumerate(param_names):
            param_idx_map[name] = start_idx + i

        # Evaluate current f values (arrays of length N_data)
        f_vals = {f_name: Z_map[f_name] @ coeffs_map[f_name] for f_name, _ in func_order}

        # Build Residuals and Jacobian blocks (now using global-index-safe filling)
        J_blocks = []
        R_blocks = []
        weights = eq_weights if isinstance(eq_weights, (list, tuple)) else [1.0] * len(equations)

        for i in range(len(equations)):
            sqrt_w = np.sqrt(weights[i]) if weights[i] is not None else 1.0

            # prepare argument array or scalars for the lambdas, in order
            lambda_args = {}
            for sym in lambdas[f'vars_{i}']:
                s_name = sym.name
                if s_name in df.columns:
                    lambda_args[s_name] = df[s_name].values
                elif s_name in f_vals:
                    lambda_args[s_name] = f_vals[s_name]
                elif s_name in scalar_vals:
                    lambda_args[s_name] = scalar_vals[s_name]
                else:
                    # default zeros if symbol not provided (should be rare)
                    lambda_args[s_name] = np.zeros(N_data)

            # Evaluate residual (array length N_data)
            try:
                # pass kwargs in symbol order
                kwargs = {s.name: lambda_args[s.name] for s in lambdas[f'vars_{i}']}
                R_i = lambdas[f'R_{i}'](**kwargs)
                R_i = np.nan_to_num(np.asarray(R_i, dtype=float)) * sqrt_w
            except (ZeroDivisionError, FloatingPointError):
                print(f"Warning: Numerical issue in iteration {iter_num}. Skipping update for equation {i}.")
                continue

            # Create a full Jacobian block for this equation (N_data x total_unknowns)
            J_i = np.zeros((N_data, total_unknowns), dtype=float)

            # derivatives wrt each function's active coefficients
            for f_name, _ in func_order:
                dRdF = lambdas[f'J_{i}_{f_name}'](**{s.name: lambda_args[s.name] for s in lambdas[f'vars_{i}']})
                dRdF_arr = np.nan_to_num(np.asarray(dRdF, dtype=float))
                if dRdF_arr.ndim == 0:
                    dRdF_arr = np.full(N_data, dRdF_arr.item(), dtype=float)
                # Z_active (N_data, n_active_f)
                Z_active = Z_map[f_name][:, coeffs_mask[f_name]]
                start_global, end_global = coeffs_idx_map[f_name]
                if Z_active.size != 0:
                    # derivative contribution to each active coefficient column:
                    # column_j = dRdF * Z_active[:, j]
                    # fill into J_i columns [start_global:end_global]
                    J_i[:, start_global:end_global] = (dRdF_arr[:, np.newaxis] * Z_active) * sqrt_w

            # derivatives wrt scalar parameters (a_i)
            for a_name in param_names:
                dRda = lambdas[f'J_{i}_{a_name}'](**{s.name: lambda_args[s.name] for s in lambdas[f'vars_{i}']})
                dRda_arr = np.nan_to_num(np.asarray(dRda, dtype=float))
                if dRda_arr.ndim == 0:
                    dRda_arr = np.full(N_data, dRda_arr.item(), dtype=float)
                col_idx = param_idx_map[a_name]
                J_i[:, col_idx] = dRda_arr * sqrt_w

            # store
            J_blocks.append(J_i)
            R_blocks.append(R_i)

        # --- Penalty residuals AND their Jacobians (fill into correct global columns) ---
        for cons in constraints:
            rule = cons.get("constraint", "")
            penalty = cons.get("penalty", 1.0)
            if not rule.startswith("f"):
                continue
            f_name_match = re.match(r'(f\d+)', rule)
            if f_name_match is None:
                continue
            f_name = f_name_match.group(1)
            if f_name not in coeffs_map:
                continue
            coeffs = coeffs_map[f_name]
            var_name = func_map[f_name]
            A0, A1 = scaling_params[var_name]
            n_coeffs = len(coeffs)
            start_global, end_global = coeffs_idx_map[f_name]
            active_powers = np.nonzero(coeffs_mask[f_name])[0]  # polynomial powers that are active
            sqrt_pen = np.sqrt(penalty)

            # f(0)=0 pointwise constraint
            if re.search(r'\(0\)\s*=\s*0', rule):
                z0 = (0.0 - A0) / A1
                val0 = np.sum([coeffs[k] * (z0**k) for k in range(n_coeffs)])
                R_pen = np.array([val0 * sqrt_pen])
                J_pen = np.zeros((1, total_unknowns), dtype=float)
                for local_idx, poly_power in enumerate(active_powers):
                    global_col = start_global + local_idx
                    J_pen[0, global_col] = (z0**poly_power) * sqrt_pen
                R_blocks.append(R_pen)
                J_blocks.append(J_pen)

            # odd/even symmetry
            elif "odd" in rule or "even" in rule:
                eval_mode = cons.get("eval", "array")
                if eval_mode == "array":
                    Nval = int(cons.get("Nval_array", 50))
                    x_test = np.linspace(-1.0, 1.0, Nval) * A1 + A0
                else:
                    x_test = df[func_map[f_name]].values
                z_test = (x_test - A0) / A1
                Z_test = np.vstack([z_test**i for i in range(n_coeffs)]).T
                z_neg = -z_test
                Z_neg = np.vstack([z_neg**i for i in range(n_coeffs)]).T
                if "odd" in rule:
                    err = (Z_test @ coeffs) + (Z_neg @ coeffs)  # should be zero for odd
                    sign = +1.0
                else:
                    err = (Z_test @ coeffs) - (Z_neg @ coeffs)  # should be zero for even
                    sign = -1.0

                R_pen = err * sqrt_pen
                M = len(R_pen)
                J_pen = np.zeros((M, total_unknowns), dtype=float)
                for local_idx, poly_power in enumerate(active_powers):
                    global_col = start_global + local_idx
                    if "odd" in rule:
                        deriv = (Z_test[:, poly_power] + Z_neg[:, poly_power])
                    else:
                        deriv = (Z_test[:, poly_power] - Z_neg[:, poly_power])
                    J_pen[:, global_col] = deriv * sqrt_pen
                R_blocks.append(R_pen)
                J_blocks.append(J_pen)

        # --- Stack everything and do Gauss-Newton step ---
        if len(J_blocks) == 0:
            print("No Jacobian blocks constructed; exiting.")
            break

        J_total = np.vstack(J_blocks)
        R_total = np.concatenate(R_blocks)
        loss = np.mean(R_total**2)

        try:
            delta_p, *_ = np.linalg.lstsq(J_total, -R_total, rcond=None)
            p += learning_rate * delta_p
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix in iteration {iter_num}. Skipping update.")

        # Logging and early stopping
        if (iter_num + 1) % 10 == 0 or loss < error_threshold:
            print(f"Iter {iter_num+1}/{n_iter}, Loss: {loss:.4e}", end="")
            if scalar_vals:
                print(", Params: ", end="")
                for name, val in scalar_vals.items():
                    print(f"{name}: {val:.3e}", end=" ")
            print()
        if loss < error_threshold:
            print(f"Early stopping at iteration {iter_num+1} as loss < {error_threshold:.1e}")
            break

    # --- Finalize ---
    final_coeffs = {}
    idx = 0
    for f_name, _ in func_order:
        mask = coeffs_mask[f_name]
        n_active = int(np.sum(mask))
        end_idx = idx + n_active
        full_coeffs = np.zeros(n_coeffs_per_func, dtype=float)
        if n_active > 0:
            full_coeffs[mask] = p[idx:end_idx]
        full_coeffs[~mask] = 0.0
        final_coeffs[f_name] = full_coeffs
        idx = end_idx

    final_scalars = {name: float(p[idx + i]) for i, name in enumerate(param_names)} if n_params > 0 else {}

    models = {}
    for f_name, var_name in func_order:
        A0, A1 = scaling_params[var_name]
        models[f_name] = {'coeffs': final_coeffs[f_name], 'A0': A0, 'A1': A1, 'var': var_name}

    # Build evals
    evals = []
    for f_name, var_name in func_order:
        model = models[f_name]
        x_data = df[var_name].values
        x_plot = np.linspace(x_data.min(), x_data.max(), 200)
        z_plot = (x_plot - model['A0']) / model['A1']
        Z_plot = np.vstack([z_plot**i for i in range(n_coeffs_per_func)]).T
        y_plot = Z_plot @ model['coeffs']
        evals.extend([x_plot, y_plot])

    print("\n--- Final Learned Parameters ---")
    for name, val in final_scalars.items():
        print(f"Learned {name}: {val:.6e}")

    return models, evals, final_scalars

