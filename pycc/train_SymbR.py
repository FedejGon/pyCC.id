"""
train_SymbR.py

Iterative symbolic-regression fitter for functions f1(x), f2(x_dot), ...
Given a list of algebraic equations like
    'x_ddot + f1(x_dot) + f2(x) - F_ext = 0'
and possibly additional equations, this script fits each f_i using PySR
in an alternating / Gauss-Seidel style loop: at each iteration we isolate one
f_i from one equation (sympy), evaluate the target using current estimates
of the other f_j's and then fit a PySR model mapping the function's variable
-> f_i(variable).

Returns: models (dict), evals (list: x_plot, y_plot pairs for each f), scalar_coefs (empty dict)

Requirements: numpy, pandas, sympy, pysr

"""

import numpy as np
import pandas as pd
import re
import sympy as sp
from collections import OrderedDict
import warnings

try:
    from pysr import PySRRegressor
except Exception as e:
    PySRRegressor = None
    warnings.warn("PySR import failed. Install pysr to use SymbR method.")


# ------------------ Utilities copied/adapted from train_polynomial ------------------
def parse_functions(equation_str):
    #pattern = r'(f\d+)\(([a-zA-Z_]+)\)' #without_numbers
    pattern = r'(f\d+)\((\w+)\)' # with numbers, i.e. f(x1)
    all_funcs = re.findall(pattern, equation_str)
    unique_funcs = list(OrderedDict.fromkeys(all_funcs))
    return unique_funcs


def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))


# ------------------ Core SymbR trainer ------------------

def train_SymbR(df, equations, params=None):
    """Fit functions f1,f2,... appearing in equations using PySR.

    Args:
        df: pandas.DataFrame with columns used in equations.
        equations: list of equation strings (each containing an '=' ).
        params: dict with keys:
            - 'pysr': dict of kwargs forwarded to PySRRegressor (niterations, unary_operators, ...)
            - 'N_fit_points': number of points to use to fit (subsample) or None
            - 'max_iterations': max outer iterations
            - 'tol': stopping tolerance on MSE change
            - 'scaling': bool whether to scale inputs to [-1,1] for fitting
    Returns:
        models: dict f_name -> {'pysr_model': PySRRegressor, 'A0':, 'A1':, 'var': var_name}
        evals: list [x_plot, y_plot, ...] pairs for plotting
        scalar_coefs: {} (kept for compatibility)
    """
    default_pysr = {
        'niterations': 100,
        'unary_operators': ['tanh'],
        'binary_operators': ['+', '-', '*'],
        'maxsize': 12,
        'populations': 10,
        'model_selection': 'best', # 'accuracy' , 'score' 
        'verbosity': 0
    }
    if params is None:
        params = {}
    #pysr_kwargs = params.get('pysr', {}) or {}
    pysr_kwargs = params.get('pysr', default_pysr)
    for key, val in default_pysr.items():
        pysr_kwargs.setdefault(key, val)
    N_fit_points = params.get('N_fit_points', 200) # none
    max_iterations = int(params.get('max_iterations',15 ))
    tol = float(params.get('tol', 1e-10))
    scaling = bool(params.get('scaling', False))
    n_eval = int(params.get('n_eval', 200))
    
    if PySRRegressor is None:
        raise ImportError('PySR is required for SymbR method. pip install pysr')

    # Prepare
    equations = list(equations) if isinstance(equations, (list, tuple)) else [equations]
    N_data = len(df)

    # Parse functions and ensure consistent usage
    func_map = OrderedDict()
    param_names = set()
    for eq in equations:
        for f_name, var_name in parse_functions(eq):
            if f_name in func_map and func_map[f_name] != var_name:
                raise ValueError(f"Function {f_name} used with different arguments.")
            func_map[f_name] = var_name
        param_names.update(extract_parameters(eq))

    func_order = list(func_map.items())

    # Scaling params per variable
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

    # Build sympy expressions parsed for algebraic manipulation
    sym_exprs = []
    for eq_str in equations:
        if '=' not in eq_str:
            raise ValueError('All equations must contain an = sign')
        lhs_str, rhs_str = eq_str.split('=', 1)
        expr = sp.sympify(lhs_str) - sp.sympify(rhs_str)
        sym_exprs.append(expr)

    # Replace f_i(var) occurrences by symbolic symbols F_i so we can isolate them algebraically
    f_syms = {f_name: sp.Symbol(f_name) for f_name, _ in func_order}
    func_call_to_sym_map = {sp.Function(f_name)(sp.Symbol(var_name)): f_syms[f_name]
                             for f_name, var_name in func_order}

    # For each equation, try to isolate each F_i if it appears in that equation
    isolate_map = {}  # (eq_index, f_name) -> sympy expression for F_i in terms of others
    for i, expr in enumerate(sym_exprs):
        expr_sub = expr.subs(func_call_to_sym_map)
        for f_name, _ in func_order:
            Fsym = f_syms[f_name]
            if Fsym in expr_sub.free_symbols:
                # Try to solve algebraically for Fsym
                try:
                    sols = sp.solve(sp.Eq(expr_sub, 0), Fsym, dict=True)
                except Exception:
                    sols = []
                if sols:
                    # pick first solution
                    sol_expr = sols[0][Fsym]
                    isolate_map[(i, f_name)] = sol_expr
                else:
                    # If not solvable, try isolating linearly by rearranging manually:
                    # gather terms: treat Fsym as linear: expr_sub = a*Fsym + rest
                    a = sp.simplify(sp.diff(expr_sub, Fsym))
                    if a == 0:
                        # can't isolate
                        continue
                    rest = expr_sub - a * Fsym
                    sol_expr = sp.simplify(-rest / a)
                    isolate_map[(i, f_name)] = sol_expr

    if not isolate_map:
        raise RuntimeError('Could not isolate any f_i from the provided equations. Check equations.')

    # Initialize storage for current f estimates (callables that accept arrays)
    current_preds = {f_name: (lambda x: np.zeros_like(x)) for f_name, _ in func_order}
    models = {f_name: None for f_name, _ in func_order}

    prev_loss = np.inf

    # Outer iterative loop: alternate between functions
    for outer in range(max_iterations):
        mse_accum = []
        for f_name, var_name in func_order:
            # find one equation that provides an isolation for this f_name
            found = False
            for (eq_i, fn), sol_expr in isolate_map.items():
                if fn != f_name:
                    continue
                # We will use this equation to compute the target for f_name
                # Build a lambda that evaluates sol_expr with dataframe columns and current preds
                # Prepare sympy free symbols and their mapping
                syms = sorted(sol_expr.free_symbols, key=lambda s: s.name)
                lam = sp.lambdify(syms, sol_expr, 'numpy')

                # Build arguments in the right order
                arg_arrays = []
                for s in syms:
                    sname = s.name
                    if sname in df.columns:
                        arr = df[sname].values
                    elif sname in f_syms:
                        # other function symbol: use current prediction evaluated at its variable
                        other_f = sname
                        var_other = func_map[other_f]
                        xvar = df[var_other].values
                        arr = current_preds[other_f](xvar)
                    else:
                        # unknown symbol: zeros
                        arr = np.zeros(N_data)
                    arg_arrays.append(arr)

                # compute target values for this f_name
                try:
                    target_vals = np.nan_to_num(np.asarray(lam(*arg_arrays), dtype=float)).ravel()
                except Exception:
                    # numerical issue evaluating lambdified expression, skip
                    continue

                # prepare training data (optionally subsample)
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

                # Build and fit PySR model for this function
                pysr_local_kwargs = pysr_kwargs.copy()
                # ensure unary/binary operators exist
                try:
                    model = PySRRegressor(**pysr_local_kwargs)
                except TypeError:
                    # backward compatibility: if some kwargs invalid, call without kwargs
                    model = PySRRegressor()

                # Fit. If X_train or y_train constant, skip and use constant model
                if np.allclose(np.std(y_train), 0.0):
                    # trivial constant function
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

                        # create prediction function (scale input inside)
                        def mk_pred(py_model, A0=A0, A1=A1):
                            def pred(x_arr):
                                z = (x_arr - A0) / A1
                                Xz = z.reshape(-1, 1)
                                # PySR returns vector
                                try:
                                    yhat = py_model.predict(Xz)
                                except Exception:
                                    # fallback: evaluate best program via .predict if available
                                    yhat = np.zeros(len(Xz))
                                return np.asarray(yhat).ravel()
                            return pred

                        current_preds[f_name] = mk_pred(model)
                        found = True
                    except Exception as e:
                        warnings.warn(f"PySR failed fitting {f_name}: {e}")
                        continue

                # compute MSE for this function's fit (full dataset)
                y_full = target_vals
                y_pred_full = current_preds[f_name](x_raw)
                mse = np.mean((y_full - y_pred_full)**2)
                mse_accum.append(mse)

                # break after using the first available isolation
                break

            if not found:
                warnings.warn(f"Could not find an isolation for {f_name} in any equation. Skipping.")

        # end loop over functions
        mean_mse = np.mean(mse_accum) if mse_accum else np.inf
        if outer % 1 == 0:
            print(f"SymbR outer iter {outer+1}/{max_iterations}, Loss: {mean_mse:.2e}")
        if abs(prev_loss - mean_mse) < tol:
            print(f"Converged with Delta_Loss={tol}. Stopping outer loop.")
            break
        prev_loss = mean_mse

    # Prepare evals for plotting
    evals = []
    for f_name, var_name in func_order:
        model = models.get(f_name, None)
        x_data = df[var_name].values
        x_plot = np.linspace(x_data.min(), x_data.max(), n_eval)
        if model is None:
            y_plot = np.zeros_like(x_plot)
        elif model.get('pysr_model') is None:
            y_plot = np.full_like(x_plot, model['const'], dtype=float)
        else:
            y_plot = current_preds[f_name](x_plot)
        evals.extend([x_plot, y_plot])

    scalar_coefs = {}
    return models, evals, scalar_coefs


## If used as script, provide a small example (guarded)
#if __name__ == '__main__':
#    # simple demonstration with synthetic data - user should import train_symb from this file.
#    print('This module provides train_symbr(df, equations, params). Import and call from your script.')

