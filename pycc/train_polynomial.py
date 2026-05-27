import numpy as np
import re
import sympy as sp
import math
from collections import OrderedDict
import copy
from scipy.optimize import least_squares
from .simulate_Poly import simulate_Poly

def parse_functions(equation_str):
    pattern = r'(f\d+)\((\w+)\)'
    all_funcs = re.findall(pattern, equation_str)
    unique_funcs = list(OrderedDict.fromkeys(all_funcs))
    return unique_funcs

def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))

def build_constraint_mask(constraints, f_name, n_coeffs):
    """
    Build a boolean mask for active polynomial coefficients,
    enforcing odd/even symmetry and f(0)=0 constraints.
    """
    # start with all True
    mask = np.ones(n_coeffs, dtype=bool)

    if not constraints:
        return mask

    for cons in constraints:
        rule = cons.get("constraint", "").strip().replace('\t', ' ')
        if not rule or not rule.startswith(f_name):
            continue

        # --- Odd symmetry ---
        if re.search(r'\bodd\b', rule):
            # only keep odd powers (1, 3, 5, ...)
            mask[:] = False
            mask[1::2] = True
            # ensure constant term is 0
            mask[0] = False

        # --- Even symmetry ---
        elif re.search(r'\beven\b', rule):
            # only keep even powers (0, 2, 4, ...)
            mask[:] = False
            mask[0::2] = True

        # --- f(0)=0 constraint ---
        if re.search(r'\(0\)\s*=\s*0', rule):
            mask[0] = False

    return mask


#def build_constraint_mask(constraints, f_name, n_coeffs):
#    mask = np.ones(n_coeffs, dtype=bool)
#    if not constraints:
#        return mask
#    for cons in constraints:
#        rule = cons.get("constraint", "").strip().replace('\t', ' ')
#        if not rule or not rule.startswith(f_name):
#            continue
#        if re.search(r'\bodd\b', rule):
#            mask[::2] = False   # zero even powers (including constant) for odd functions
#        elif re.search(r'\beven\b', rule):
#            mask[1::2] = False  # zero odd powers for even functions
#        elif re.search(r'\(0\)\s*=\s*0', rule):
#            mask[0] = False
#    return mask

# this function performs forward simulations and updates the model 
#def simulation_residuals(p_flat, df, eqs, base_models, base_scalars, state_vars, params_simul_dict):

def simulation_residuals(p_flat, df, eqs, base_models, base_scalars, state_vars, params_simul_dict, param_names, tracker):    
    """
    Objective function for the simulation-based outer loop.
    Unpacks parameters, runs the forward simulation, and returns the residuals against df.
    """
    # 1. Unpack flat parameter array back into dictionaries
    idx = 0
    new_models = copy.deepcopy(base_models)
    # We rely on base_models keeping insertion order (dict in Python 3.7+ keeps order)
    for f_name in base_models.keys():
        n_coeffs = len(new_models[f_name]['coeffs'])
        new_models[f_name]['coeffs'] = p_flat[idx : idx + n_coeffs]
        idx += n_coeffs
        
    new_scalars = copy.deepcopy(base_scalars)
    for a_name in base_scalars.keys():
        new_scalars[a_name] = p_flat[idx]
        idx += 1

    # 2. Setup simulation parameters
    sim_kwargs = copy.deepcopy(params_simul_dict)
    sim_kwargs['models'] = new_models
    sim_kwargs['obtained_coefs'] = new_scalars
    t_data = sim_kwargs.get('t_eval')

    # 3. Run simulation
    try:
        sol, _ = simulate_Poly(eqs, sim_kwargs)
        
        # Penalize if integration failed or didn't finish
        if sol.status != 0 or len(sol.t) != len(t_data):
            return np.ones(len(t_data) * len(state_vars)) * 1e6
            
    except Exception as e:
        return np.ones(len(t_data) * len(state_vars)) * 1e6

    # 4. Compute residuals (Simulated States - Actual Database States)
    residuals = []
    for i, var in enumerate(state_vars):
        simulated_trajectory = sol.y[i, :]
        actual_trajectory = df[var].values
        residuals.append(simulated_trajectory - actual_trajectory)

    residuals_concat = np.concatenate(residuals)

    # Custom Logging ---
    tracker['count'] += 1
    loss = np.mean(residuals_concat**2)
    
    # Format the scalar parameters nicely
    param_str = " ".join([f"{name}: {new_scalars[name]:.3e}" for name in param_names if name in new_scalars])
    max_iter = tracker['max_iter'] if tracker['max_iter'] is not None else "inf"
    
    # Print the log line
    print(f"Iter {tracker['count']}/{max_iter}, Loss_x: {loss:.4e}, Params: {param_str}")

    return residuals_concat


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
    n_eval = int(params.get('n_eval', 200))
    print_poly_coeffs = bool(params.get('print_poly_coeffs', True))
    initial_param_guess = params.get('initial_param_guess', [])
    init_guesses = {}
    for d in initial_param_guess:
        init_guesses.update(d)
    fitting_forw_sim = bool(params.get('fitting_forw_sim', False))
    run_sniper = bool(params.get('run_sniper', False))
    diff_step = float(params.get('diff_step', 1e-2))
    n_iter_outer = params.get('n_iter_outer', None)   # Defaults to None (letting scipy decide)
    outer_tol = float(params.get('outer_tol', 1e-8))  # Default tolerance outer loop
    params_simul_list = params.get('params_simul', [])
    params_simul_dict = {}
    for d in params_simul_list:
        params_simul_dict.update(d)
        
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

    # Apply initial guesses
    for i, a_name in enumerate(param_names):
        if a_name in init_guesses:
            # Scalar params are located at the end of p, starting at index n_active_coeffs_total
            p_idx = n_active_coeffs_total + i
            p[p_idx] = float(init_guesses[a_name])
            print(f"Initialized parameter '{a_name}' to {init_guesses[a_name]}")

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
        scalar_slice = p[start_idx:start_idx + n_params] if n_params > 0 else np.array([])
        scalar_vals = {name: float(scalar_slice[i]) for i, name in enumerate(param_names)} if n_params > 0 else {}
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
                    lambda_args[s_name] = np.zeros(N_data)

            # Evaluate residual (array length N_data)
            try:
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
                Z_active = Z_map[f_name][:, coeffs_mask[f_name]]
                start_global, end_global = coeffs_idx_map[f_name]
                if Z_active.size != 0:
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
        # We support:
        #  - point constraints: f1(x0)=y0  (generic)
        #  - special case f1(0)=0 (also covered by the generic)
        #  - odd / even symmetry constraints with eval='array' or 'data'
        float_re = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        point_pat = re.compile(r'^(f\d+)\(\s*' + float_re + r'\s*\)\s*=\s*' + float_re + r'\s*$', re.I)

        for cons in constraints:
            rule = cons.get("constraint", "").strip()
            penalty = float(cons.get("penalty", 1.0))
            eval_mode = cons.get("eval", "array")
            sqrt_pen = np.sqrt(penalty)

            if not rule:
                continue

            # --- Generic value-at-point constraint: fN(x0)=y0 ---
            m_val = point_pat.match(rule)
            if m_val:
                f_name = m_val.group(1)
                x_val = float(m_val.group(2))
                y_target = float(m_val.group(3))
                if f_name not in coeffs_map:
                    continue
                coeffs = coeffs_map[f_name]
                var_name = func_map[f_name]
                A0, A1 = scaling_params[var_name]
                # compute z at the point
                z0 = (x_val - A0) / A1
                # build Z_point vector (length n_coeffs_per_func)
                Z_point = np.array([z0**k for k in range(n_coeffs_per_func)], dtype=float)
                # residual is (Z_point @ coeffs - y_target) * sqrt_pen
                val0 = float(np.dot(Z_point, coeffs) - y_target)
                R_pen = np.array([val0 * sqrt_pen])
                J_pen = np.zeros((1, total_unknowns), dtype=float)
                active_powers = np.nonzero(coeffs_mask[f_name])[0]
                for local_idx, poly_power in enumerate(active_powers):
                    global_col = coeffs_idx_map[f_name][0] + local_idx
                    J_pen[0, global_col] = (z0**poly_power) * sqrt_pen
                # no derivatives w.r.t scalar parameters in this point-constraint (unless you'd want to)
                R_blocks.append(R_pen)
                J_blocks.append(J_pen)
                # done with this constraint
                continue

            # --- f(0)=0 legacy style (if not captured above) ---
            m_f0 = re.match(r'^(f\d+)\(\s*0\s*\)\s*=\s*0\s*$', rule)
            if m_f0:
                f_name = m_f0.group(1)
                if f_name not in coeffs_map:
                    continue
                coeffs = coeffs_map[f_name]
                var_name = func_map[f_name]
                A0, A1 = scaling_params[var_name]
                n_coeffs = len(coeffs)
                start_global, end_global = coeffs_idx_map[f_name]
                # evaluate function at x=0 (scaled z0)
                z0 = (0.0 - A0) / A1
                val0 = np.sum([coeffs[k] * (z0**k) for k in range(n_coeffs)])
                R_pen = np.array([val0 * sqrt_pen])
                J_pen = np.zeros((1, total_unknowns), dtype=float)
                active_powers = np.nonzero(coeffs_mask[f_name])[0]
                for local_idx, poly_power in enumerate(active_powers):
                    global_col = start_global + local_idx
                    J_pen[0, global_col] = (z0**poly_power) * sqrt_pen
                R_blocks.append(R_pen)
                J_blocks.append(J_pen)
                continue

            # --- odd/even symmetry ---
            if re.search(r'\bodd\b', rule, re.I) or re.search(r'\beven\b', rule, re.I):
                # check which f this constraint refers to: must start with fN
                f_name_match = re.match(r'^(f\d+)', rule)
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
                active_powers = np.nonzero(coeffs_mask[f_name])[0]

                # build evaluation x values
                if eval_mode == "array":
                    Nval = int(cons.get("Nval_array", 50))
                    x_test = np.linspace(-1.0, 1.0, Nval) * A1 + A0
                else:
                    # 'data' or anything else -> use actual df points of this variable
                    x_test = df[func_map[f_name]].values
                z_test = (x_test - A0) / A1
                Z_test = np.vstack([z_test**i for i in range(n_coeffs)]).T
                z_neg = -z_test
                Z_neg = np.vstack([z_neg**i for i in range(n_coeffs)]).T

                if re.search(r'\bodd\b', rule, re.I):
                    err = (Z_test @ coeffs) + (Z_neg @ coeffs)  # should be zero for odd
                    # derivative w.r.t active coeffs: Z_test[:,p] + Z_neg[:,p]
                    M = len(err)
                    R_pen = err * sqrt_pen
                    J_pen = np.zeros((M, total_unknowns), dtype=float)
                    for local_idx, poly_power in enumerate(active_powers):
                        global_col = start_global + local_idx
                        deriv = (Z_test[:, poly_power] + Z_neg[:, poly_power])
                        J_pen[:, global_col] = deriv * sqrt_pen
                    R_blocks.append(R_pen)
                    J_blocks.append(J_pen)
                else:
                    # even
                    err = (Z_test @ coeffs) - (Z_neg @ coeffs)  # should be zero for even
                    M = len(err)
                    R_pen = err * sqrt_pen
                    J_pen = np.zeros((M, total_unknowns), dtype=float)
                    for local_idx, poly_power in enumerate(active_powers):
                        global_col = start_global + local_idx
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
        if (iter_num + 1) % 100 == 0 or loss < error_threshold:
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
        
        
    if fitting_forw_sim:
        print("\n--- Starting Forward Simulation Fine-Tuning ---")
        # Extract state variables from equations (e.g., 'x1_dot = ...' -> 'x1')
        state_vars = [eq.split('_dot')[0].strip() for eq in equations]
        
        # Flatten the current best guess into p0 array
        p0 = []
        for f_name, _ in func_order: 
            p0.extend(models[f_name]['coeffs'])
        for a_name in param_names:
            p0.append(final_scalars[a_name])
        p0 = np.array(p0)
        
        tracker = {'count': 0, 'max_iter': n_iter_outer}

        # ==========================================
        # PHASE 1: The Brawler (Your original aggressive settings)
        # ==========================================
        if run_sniper:
            print("\nPhase 1: Global search with large diff_step and soft_l1 loss...")
            
        res = least_squares(
            simulation_residuals, 
            x0=p0, 
            args=(df, equations, models, final_scalars, state_vars, params_simul_dict, param_names, tracker),
            method='trf',            
            x_scale='jac',          
            diff_step=diff_step,         
            loss='soft_l1',         
            max_nfev=n_iter_outer,
            ftol=outer_tol * 1e-4,  
            xtol=outer_tol * 1e-4,
            gtol=outer_tol * 1e-4,
            verbose=0
        )
        
        if run_sniper:
            print(f"Phase 1 Finished. Success: {res.success}, Evals: {res.nfev}")

            # ==========================================
            # PHASE 2: The Sniper (Local Refinement)
            # ==========================================
            print("\nPhase 2: Local refinement dropping into the basin...")
            tracker['count'] = 0 
            
            p0_phase2 = res.x

            res_phase2 = least_squares(
                simulation_residuals, 
                x0=p0_phase2, 
                args=(df, equations, models, final_scalars, state_vars, params_simul_dict, param_names, tracker),
                method='trf',             
                x_scale='jac',            
                diff_step=diff_step*1e-4, # Small step to find the exact minimum
                loss='linear',            # Standard least squares
                max_nfev=n_iter_outer,
                ftol=outer_tol* 1e-4,           
                xtol=outer_tol* 1e-4,
                gtol=outer_tol* 1e-4,
                verbose=0
            )
            
            print(f"\nPhase 2 Fine-Tuning Finished. Success: {res_phase2.success}, Evals: {res_phase2.nfev}")
            res = res_phase2 # Overwrite res so the unpacking below works seamlessly
            
        else:
            print(f"\nSimulation Fine-Tuning Finished. Success: {res.success}")

        # ==========================================
        # Unpack Results (Applies to whichever phase finished last)
        # ==========================================
        idx = 0
        for f_name, _ in func_order:
            n_coeffs = len(models[f_name]['coeffs'])
            models[f_name]['coeffs'] = res.x[idx : idx + n_coeffs]
            idx += n_coeffs
            
        for a_name in param_names:
            final_scalars[a_name] = res.x[idx]
            idx += 1

#        # Run Trust Region Reflective optimizer with aggressive settings
#        res = least_squares(
#            simulation_residuals, 
#            x0=p0, 
#            args=(df, equations, models, final_scalars, state_vars, params_simul_dict, param_names, tracker),
#            method='trf',           # Switch to Trust Region Reflective
#            x_scale='jac',          # Crucial: Auto-scales parameters based on the Jacobian so a1 and a2 are treated fairly
#            diff_step=1e-2,         # Violent step: Forces the gradient calculation to look much wider (default is ~1e-8)
#            loss='soft_l1',         # Makes it more robust to sudden large errors during simulation
#            max_nfev=n_iter_outer,
#            ftol=outer_tol * 1e-4,  # Crush the tolerances to force it to keep trying
#            xtol=outer_tol * 1e-4,
#            gtol=outer_tol * 1e-4,
#            verbose=0
#        )
#        
#
#        # Run Levenberg-Marquardt optimizer
#        res = least_squares(
#            simulation_residuals, 
#            x0=p0, 
#            #args=(df, equations, models, final_scalars, state_vars, params_simul_dict),
#            args=(df, equations, models, final_scalars, state_vars, params_simul_dict, param_names, tracker),
#            method='lm',
#            max_nfev=n_iter_outer,   # Limits the maximum iterations
#            ftol=outer_tol,          # Cost function tolerance
#            xtol=outer_tol,          # Step size tolerance
#            gtol=outer_tol,          # Gradient tolerance
#            verbose=2
#        )        
#        
#        print("\nSimulation Fine-Tuning Finished. Success:", res.success)
#        
#        # Overwrite models and final_scalars with the newly optimized values
#        idx = 0
#        for f_name, _ in func_order:
#            n_coeffs = len(models[f_name]['coeffs'])
#            models[f_name]['coeffs'] = res.x[idx : idx + n_coeffs]
#            idx += n_coeffs
#            
#        for a_name in param_names:
#            final_scalars[a_name] = res.x[idx]
#            idx += 1    



    # Build evals
    evals = []
    for f_name, var_name in func_order:
        model = models[f_name]
        x_data = df[var_name].values
        x_plot = np.linspace(x_data.min(), x_data.max(), n_eval)
        z_plot = (x_plot - model['A0']) / model['A1']
        Z_plot = np.vstack([z_plot**i for i in range(n_coeffs_per_func)]).T
        y_plot = Z_plot @ model['coeffs']
        evals.extend([x_plot, y_plot])

    print("\n--- Final Learned Parameters ---")
    for name, val in final_scalars.items():
        print(f"Learned {name}: {val:.6e}")


    # Print scaled and unscaled polynomial coefficients
    if print_poly_coeffs:
        print("\n--- Final Polynomial Coefficients ---")
        for f_name, var_name in func_order:
            model = models[f_name]
            coeffs = model['coeffs']
            A0 = model['A0']
            A1 = model['A1']
            
            print(f"Function {f_name}({var_name}):")
            print(f"  Scaled Coeffs (w.r.t z = (x - {A0:.4e}) / {A1:.4e}):")
            for i, c in enumerate(coeffs):
                # Optionally hide very small terms to keep the console clean
                if abs(c) > 1e-12: 
                    print(f"    z^{i}: {c:.6e}")
            
            # Unscale the coefficients
            unscaled_coeffs = np.zeros_like(coeffs)
            N_len = len(coeffs) - 1
            for i in range(N_len + 1):
                Ci = coeffs[i] / (A1**i)
                for k in range(i + 1):
                    # Uses the binomial theorem to expand the shifted term
                    unscaled_coeffs[k] += Ci * math.comb(i, k) * ((-A0)**(i - k))
                    
            print(f"  Unscaled Coeffs (w.r.t original {var_name}):")
            for k, c in enumerate(unscaled_coeffs):
                if abs(c) > 1e-12:
                    print(f"    {var_name}^{k}: {c:.6e}")
            print()
    

    return models, evals, final_scalars

