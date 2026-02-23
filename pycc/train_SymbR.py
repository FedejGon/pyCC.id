"""
train_SymbR.py

Iterative symbolic-regression fitter for functions f1(x), f2(x_dot), ...
Given a list of algebraic equations like
    'x_ddot + f1(x_dot) + f2(x) - F_ext = 0'
and possibly additional equations, this script fits each f_i using PySR
in an alternating / Gauss-Seidel style loop.

Optionally performs forward-simulation fine-tuning of the discovered constants.

Returns: models (dict), evals (list: x_plot, y_plot pairs for each f), scalar_coefs (empty dict)
"""

import numpy as np
import pandas as pd
import re
import sympy as sp
from collections import OrderedDict
import warnings

# For fine-tuning
from scipy.optimize import minimize
try:
    from .simulate_SymbR import simulate_SymbR
except ImportError:
    simulate_SymbR = None

try:
    from pysr import PySRRegressor
except Exception as e:
    PySRRegressor = None
    warnings.warn("PySR import failed. Install pysr to use SymbR method.")


# ------------------ Utilities ------------------
def parse_functions(equation_str):
    pattern = r'(f\d+)\((\w+)\)' 
    all_funcs = re.findall(pattern, equation_str)
    unique_funcs = list(OrderedDict.fromkeys(all_funcs))
    return unique_funcs


def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))


# ------------------ Fine-Tuning Module ------------------


def _fine_tune_constants(models, df, equations, params, func_order):
    """
    Extracts constants from PySR symbolic expressions and optimizes them 
    using forward simulation to minimize trajectory MSE.
    Prints intermediate losses during the iterative procedure.
    """
    if simulate_SymbR is None:
        warnings.warn("simulate_SymbR not found. Skipping fine-tuning.")
        return models, {}
    
    print("Starting forward-simulation fine-tuning...")
    
    # Parse params_simul (handles list of dicts or single dict)
    sim_params_raw = params.get('params_simul', {})
    sim_params = {}
    if isinstance(sim_params_raw, list):
        for d in sim_params_raw:
            sim_params.update(d)
    else:
        sim_params = sim_params_raw.copy()

    # Determine ground truth trajectories from dataframe
    state_vars = []
    for eq in equations:
        if '=' in eq:
            lhs = eq.split('=')[0].strip()
            state_vars.append(lhs.replace('_dot', ''))
            
    try:
        true_traj = np.vstack([df[sv].values for sv in state_vars])
    except KeyError as e:
        warnings.warn(f"State variable missing from dataframe for fine-tuning: {e}")
        return models, {}

    n_iter_outer = params.get('n_iter_outer', 200)
    outer_tol = params.get('outer_tol', 1e-6)

    # Extract expressions and map constants
    flat_params0 = []
    param_mappings = {} # f_name -> (sym_expr, list_of_param_syms)
    x0_sym = sp.Symbol('x0')
    
    for f_name, m in models.items():
        pm = m.get('pysr_model')
        if pm is not None:
            try:
                expr = pm.sympy()
            except Exception:
                expr = sp.sympify(str(pm.get_best()['equation']))
            
            floats = list(expr.atoms(sp.Float))
            c_syms = [sp.Symbol(f'c_{f_name}_{i}') for i in range(len(floats))]
            
            subs_dict = dict(zip(floats, c_syms))
            expr_param = expr.subs(subs_dict)
            
            param_mappings[f_name] = (expr_param, c_syms)
            flat_params0.extend([float(fl) for fl in floats])
        else:
            val = m.get('const', 0.0)
            c_sym = sp.Symbol(f'c_{f_name}_0')
            param_mappings[f_name] = (c_sym, [c_sym])
            flat_params0.append(float(val))
            
    flat_params0 = np.array(flat_params0)
    
    if len(flat_params0) == 0:
        print("No floating-point constants found to fine-tune.")
        return models, {}

    # --- Tracking variables for intermediate printing ---
    eval_counter = 0
    best_loss = np.inf

    def objective(p_array):
        nonlocal eval_counter, best_loss
        eval_counter += 1
        
        idx = 0
        temp_models = {}
        for f_name, m in models.items():
            expr_param, c_syms = param_mappings[f_name]
            n_c = len(c_syms)
            p_vals = p_array[idx:idx+n_c]
            idx += n_c
            
            subs_dict = dict(zip(c_syms, p_vals))
            new_expr = expr_param.subs(subs_dict)
            
            var_name = m['var']
            A0, A1 = m['A0'], m['A1']
            
            if not new_expr.free_symbols.difference(set(c_syms)):
                const_val = float(new_expr)
                temp_models[f_name] = {'const': const_val, 'A0': A0, 'A1': A1, 'var': var_name, 'expr': new_expr}
            else:
                sym = x0_sym if x0_sym in new_expr.free_symbols else list(new_expr.free_symbols)[0]
                lam = sp.lambdify(sym, new_expr, 'numpy')
                
                def mk_pred(lam_func, a0=A0, a1=A1):
                    def pred(x_arr):
                        z = (np.asarray(x_arr) - a0) / a1
                        res = lam_func(z)
                        if np.isscalar(res):
                            return np.full_like(z, res, dtype=float)
                        return np.asarray(res, dtype=float)
                    return pred
                
                temp_models[f_name] = {'func': mk_pred(lam), 'var': var_name, 'A0': A0, 'A1': A1, 'expr': new_expr}
                
        sim_p = sim_params.copy()
        sim_p['models'] = temp_models
        sim_p['print_models'] = False
        sim_p['check_nan'] = False
        
        try:
            sol, _ = simulate_SymbR(equations, sim_p)
            if sol.y.shape != true_traj.shape:
                mse = 1e6 # Penalty for shape mismatch
            else:
                mse = np.mean((sol.y - true_traj)**2)
        except Exception:
            mse = 1e6 # Penalty for divergent integration

        # --- Print intermediate results ---
        if mse < best_loss and mse != 1e6:
            best_loss = mse
            print(f"  -> Eval {eval_counter:03d} | New best MSE: {best_loss:.6e}")
        elif eval_counter % 20 == 0:
            print(f"  -> Eval {eval_counter:03d} | Current MSE: {mse:.6e} (Best: {best_loss:.6e})")

        return mse

    # Optimize
    print(f"Optimizing {len(flat_params0)} parameters...")
    res = minimize(objective, flat_params0, method='Nelder-Mead', 
                   options={'maxiter': n_iter_outer, 'xatol': outer_tol, 'disp': True})
    
    print(f"Fine-tuning complete. Final MSE: {res.fun:.6e}")
    
    # Rebuild final updated models dict & prediction closures
    idx = 0
    updated_preds = {}
    for f_name, m in models.items():
        expr_param, c_syms = param_mappings[f_name]
        n_c = len(c_syms)
        best_p = res.x[idx:idx+n_c]
        idx += n_c
        
        best_expr = expr_param.subs(dict(zip(c_syms, best_p)))
        models[f_name]['expr'] = best_expr
        
        A0, A1 = m['A0'], m['A1']
        if not best_expr.free_symbols.difference(set(c_syms)):
            const_val = float(best_expr)
            updated_preds[f_name] = lambda x, c=const_val: np.full_like(x, c, dtype=float)
            models[f_name]['const'] = const_val
            models[f_name]['pysr_model'] = None 
        else:
            sym = x0_sym if x0_sym in best_expr.free_symbols else list(best_expr.free_symbols)[0]
            lam = sp.lambdify(sym, best_expr, 'numpy')
            def mk_pred(lam_func, a0=A0, a1=A1):
                def pred(x_arr):
                    z = (np.asarray(x_arr) - a0) / a1
                    res = lam_func(z)
                    if np.isscalar(res):
                        return np.full_like(z, res, dtype=float)
                    return np.asarray(res, dtype=float)
                return pred
            
            updated_preds[f_name] = mk_pred(lam)
            models[f_name]['func'] = updated_preds[f_name]

    return models, updated_preds


# ------------------ Core SymbR trainer ------------------

def train_SymbR(df, equations, params=None):
    """Fit functions f1,f2,... appearing in equations using PySR."""
    # ... [Same Default Setup as before] ...
    default_pysr = {
        'niterations': 100,
        'unary_operators': ['tanh'],
        'binary_operators': ['+', '-', '*'],
        'maxsize': 12,
        'populations': 10,
        'model_selection': 'best', 
        'verbosity': 0
    }
    if params is None:
        params = {}
    pysr_kwargs = params.get('pysr', default_pysr)
    for key, val in default_pysr.items():
        pysr_kwargs.setdefault(key, val)
    N_fit_points = params.get('N_fit_points', 200)
    max_iterations = int(params.get('max_iterations',15 ))
    tol = float(params.get('tol', 1e-10))
    scaling = bool(params.get('scaling', False))
    n_eval = int(params.get('n_eval', 200))
    
    if PySRRegressor is None:
        raise ImportError('PySR is required for SymbR method. pip install pysr')

    equations = list(equations) if isinstance(equations, (list, tuple)) else [equations]
    N_data = len(df)

    func_map = OrderedDict()
    param_names = set()
    for eq in equations:
        for f_name, var_name in parse_functions(eq):
            if f_name in func_map and func_map[f_name] != var_name:
                raise ValueError(f"Function {f_name} used with different arguments.")
            func_map[f_name] = var_name
        param_names.update(extract_parameters(eq))

    func_order = list(func_map.items())

    scaling_params = {}
    for _, var_name in func_order:
        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' used in functions not found in dataframe columns")
        x = df[var_name].values.astype(float)
        if scaling:
            A0 = float((np.max(x) + np.min(x)) / 2.0)
            A1 = float((np.max(x) - np.min(x)) / 2.0)
            if A1 == 0.0:
                A1 = 1.0
        else:
            A0, A1 = 0.0, 1.0
        scaling_params[var_name] = (A0, A1)

    sym_exprs = []
    for eq_str in equations:
        if '=' not in eq_str:
            raise ValueError('All equations must contain an = sign')
        lhs_str, rhs_str = eq_str.split('=', 1)
        expr = sp.sympify(lhs_str) - sp.sympify(rhs_str)
        sym_exprs.append(expr)

    f_syms = {f_name: sp.Symbol(f_name) for f_name, _ in func_order}
    func_call_to_sym_map = {sp.Function(f_name)(sp.Symbol(var_name)): f_syms[f_name]
                             for f_name, var_name in func_order}

    isolate_map = {}
    for i, expr in enumerate(sym_exprs):
        expr_sub = expr.subs(func_call_to_sym_map)
        for f_name, _ in func_order:
            Fsym = f_syms[f_name]
            if Fsym in expr_sub.free_symbols:
                try:
                    sols = sp.solve(sp.Eq(expr_sub, 0), Fsym, dict=True)
                except Exception:
                    sols = []
                if sols:
                    sol_expr = sols[0][Fsym]
                    isolate_map[(i, f_name)] = sol_expr
                else:
                    a = sp.simplify(sp.diff(expr_sub, Fsym))
                    if a == 0:
                        continue
                    rest = expr_sub - a * Fsym
                    sol_expr = sp.simplify(-rest / a)
                    isolate_map[(i, f_name)] = sol_expr

    if not isolate_map:
        raise RuntimeError('Could not isolate any f_i from the provided equations.')

    current_preds = {f_name: (lambda x: np.zeros_like(x)) for f_name, _ in func_order}
    models = {f_name: None for f_name, _ in func_order}

    prev_loss = np.inf

    # ... [Same Outer Iterative Loop as before] ...
    for outer in range(max_iterations):
        mse_accum = []
        for f_name, var_name in func_order:
            found = False
            for (eq_i, fn), sol_expr in isolate_map.items():
                if fn != f_name:
                    continue
                syms = sorted(sol_expr.free_symbols, key=lambda s: s.name)
                lam = sp.lambdify(syms, sol_expr, 'numpy')

                arg_arrays = []
                for s in syms:
                    sname = s.name
                    if sname in df.columns:
                        arr = df[sname].values
                    elif sname in f_syms:
                        other_f = sname
                        var_other = func_map[other_f]
                        xvar = df[var_other].values
                        arr = current_preds[other_f](xvar)
                    else:
                        arr = np.np.zeros(N_data)
                    arg_arrays.append(arr)

                try:
                    target_vals = np.nan_to_num(np.asarray(lam(*arg_arrays), dtype=float)).ravel()
                except Exception:
                    continue

                x_raw = df[var_name].values.astype(float)
                A0, A1 = scaling_params[var_name]
                x_scaled = (x_raw - A0) / A1

                if N_fit_points is not None and N_fit_points < len(x_scaled):
                    idx = np.linspace(0, len(x_scaled)-1, N_fit_points, dtype=int)
                    X_train = x_scaled[idx].reshape(-1, 1)
                    y_train = target_vals[idx]
                else:
                    X_train = x_scaled.reshape(-1, 1)
                    y_train = target_vals

                pysr_local_kwargs = pysr_kwargs.copy()
                try:
                    model = PySRRegressor(**pysr_local_kwargs)
                except TypeError:
                    model = PySRRegressor()

                if np.allclose(np.std(y_train), 0.0):
                    const_val = float(np.mean(y_train))
                    def const_pred(x, c=const_val):
                        return np.full_like(x, c, dtype=float)
                    models[f_name] = {'pysr_model': None, 'A0': A0, 'A1': A1, 'var': var_name, 'const': const_val}
                    current_preds[f_name] = lambda xx, c=const_val: np.full_like(xx, c, dtype=float)
                    found = True
                else:
                    try:
                        model.fit(X_train, y_train)
                        models[f_name] = {'pysr_model': model, 'A0': A0, 'A1': A1, 'var': var_name}

                        def mk_pred(py_model, A0=A0, A1=A1):
                            def pred(x_arr):
                                z = (x_arr - A0) / A1
                                Xz = z.reshape(-1, 1)
                                try:
                                    yhat = py_model.predict(Xz)
                                except Exception:
                                    yhat = np.zeros(len(Xz))
                                return np.asarray(yhat).ravel()
                            return pred

                        current_preds[f_name] = mk_pred(model)
                        found = True
                    except Exception as e:
                        warnings.warn(f"PySR failed fitting {f_name}: {e}")
                        continue

                y_full = target_vals
                y_pred_full = current_preds[f_name](x_raw)
                mse = np.mean((y_full - y_pred_full)**2)
                mse_accum.append(mse)
                break

            if not found:
                warnings.warn(f"Could not find an isolation for {f_name} in any equation. Skipping.")

        mean_mse = np.mean(mse_accum) if mse_accum else np.inf
        if outer % 1 == 0:
            print(f"SymbR outer loop. Iter {outer+1}/{max_iterations}, Loss: {mean_mse:.2e}")
        if abs(prev_loss - mean_mse) < tol:
            print(f"Converged with Delta_Loss={tol}. Stopping outer loop.")
            break
        prev_loss = mean_mse

    # --- NEW: Forward-Simulation Fine-Tuning Block ---
    if params.get('fitting_forw_sim', False):
        models, updated_preds = _fine_tune_constants(models, df, equations, params, func_order)
        # Update current_preds with the optimized versions so `evals` generation uses them
        for fn_name, updated_fn in updated_preds.items():
            current_preds[fn_name] = updated_fn

    # Prepare evals for plotting
    evals = []
    for f_name, var_name in func_order:
        model = models.get(f_name, None)
        x_data = df[var_name].values
        x_plot = np.linspace(x_data.min(), x_data.max(), n_eval)
        if model is None:
            y_plot = np.zeros_like(x_plot)
        else:
            y_plot = current_preds[f_name](x_plot)
        evals.extend([x_plot, y_plot])

    scalar_coefs = {}
    return models, evals, scalar_coefs
