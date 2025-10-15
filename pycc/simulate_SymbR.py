"""
simulate_SymbR.py

Forward simulator compatible with the output of train_SymbR.train.

Provides `simulate_SymbR(equations, params)` that performs an ODE integration
of algebraic equations of the form 'x1_dot = ...', 'x2_dot = ...' where f1,f2
are symbolic/regression models returned by train_SymbR.

Expected params keys (required):
  - 'models' : dict returned by train_SymbR (f_name -> dict with keys 'pysr_model' or 'const', 'A0','A1','var')
  - 't_span'  : (t0, tf)
  - 'y0'      : initial state vector (ordered according to equations)
Optional:
  - 'obtained_coefs' or 'scalar_params' : dict of scalar coefficients (a1,a2,...)
  - 'local_funcs' : dict of user-provided callables (e.g. F_ext(t) or other functions)
  - 't_eval', 'method', 'atol', 'rtol', 'check_nan'

Returns: (sol, derivatives_array)
  - sol: SolveIVP result
  - derivatives_array: shape (n_states, len(sol.t)) array with evaluated RHS at each time

"""

import warnings
from typing import Dict, Any, List, Tuple, Callable
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp


def _parse_equation(eq: str) -> Tuple[str, sp.Expr]:
    s = eq.replace("^", "**")
    if '=' not in s:
        raise ValueError("Equation must contain '='")
    lhs, rhs = s.split('=', 1)
    lhs_name = lhs.strip()
    rhs_expr = sp.sympify(rhs, evaluate=False)
    return lhs_name, rhs_expr


def _make_model_pred(model_dict: Dict[str, Any]):
    """Return a callable pred(x_array)->array for a trained model dict.

    model_dict expected keys:
      - 'pysr_model' : PySRRegressor instance or None
      - 'A0','A1' : floats for input scaling
      - OR 'const' : constant value
    """
    if model_dict is None:
        return lambda x: np.zeros_like(x, dtype=float)

    # constant model
    if model_dict.get('pysr_model') is None:
        const_val = float(model_dict.get('const', 0.0))

        def const_pred(x_arr, c=const_val):
            x_arr = np.asarray(x_arr)
            return np.full_like(x_arr, c, dtype=float)

        return const_pred

    py_model = model_dict.get('pysr_model')
    A0 = float(model_dict.get('A0', 0.0))
    A1 = float(model_dict.get('A1', 1.0))

    def pred(x_arr):
        x_arr = np.asarray(x_arr, dtype=float)
        # ensure 1d
        z = (x_arr - A0) / A1
        Xz = z.reshape(-1, 1)
        try:
            yhat = py_model.predict(Xz)
        except Exception as e:
            warnings.warn(f"PySR model prediction failed: {e}")
            # fallback: zeros
            yhat = np.zeros(len(Xz))
        return np.asarray(yhat).ravel()

    return pred


def _eval_sympy_symbr(expr: sp.Basic,
                      var_values: Dict[str, float],
                      t: float,
                      scalar_params: Dict[str, float],
                      local_funcs: Dict[str, Callable],
                      model_preds: Dict[str, Callable]):
    """Evaluate a sympy expression using provided maps.

    Differences to a pure theoretical evaluator:
      - if a function name (e.g. 'f1') appears and is present in model_preds,
        the model is called with the evaluated argument value(s).
      - otherwise use local_funcs or numpy math functions.
    """
    # Number
    if expr.is_Number:
        return float(expr)

    # Symbol
    if expr.is_Symbol:
        name = str(expr)
        if name in var_values:
            return float(var_values[name])
        if name in scalar_params:
            return float(scalar_params[name])
        if name in local_funcs:
            # bare symbol treated as a time-dependent 0-arg function
            try:
                return float(local_funcs[name](t))
            except Exception as e:
                raise RuntimeError(f"Calling local_funcs['{name}'] failed: {e}")
        raise ValueError(f"Unknown symbol '{name}' at t={t}")

    # Function call
    if expr.is_Function:
        fname = expr.func.__name__
        # evaluate args recursively
        arg_vals = [_eval_sympy_symbr(a, var_values, t, scalar_params, local_funcs, model_preds)
                    for a in expr.args]

        # If this function is one of the trained models (f1,f2,...)
        if fname in model_preds:
            # expect single-argument functions f(x)
            if len(arg_vals) != 1:
                raise ValueError(f"Model function '{fname}' expected 1 arg, got {len(arg_vals)}")
            xval = np.asarray([arg_vals[0]], dtype=float)
            try:
                y = model_preds[fname](xval)
                return float(y.ravel()[0])
            except Exception as e:
                raise RuntimeError(f"Model '{fname}' prediction failed for input {xval}: {e}")

        # then check user-supplied local functions (e.g. F_ext)
        if fname in local_funcs:
            func = local_funcs[fname]
            try:
                return float(func(*arg_vals))
            except TypeError:
                # maybe user expects time only
                try:
                    return float(func(t))
                except Exception as e:
                    raise RuntimeError(f"local_funcs['{fname}'] call failed: {e}")
            except Exception as e:
                raise RuntimeError(f"local_funcs['{fname}'] raised: {e}")

        # built-in math functions -> numpy
        numpy_map = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': abs
        }
        if fname in numpy_map:
            return float(numpy_map[fname](arg_vals[0]))

        raise ValueError(f"Unknown function '{fname}' in expression")

    # Add
    if expr.is_Add:
        return float(sum(_eval_sympy_symbr(a, var_values, t, scalar_params, local_funcs, model_preds) for a in expr.args))

    # Mul
    if expr.is_Mul:
        prod = 1.0
        for a in expr.args:
            prod *= _eval_sympy_symbr(a, var_values, t, scalar_params, local_funcs, model_preds)
        return float(prod)

    # Pow
    if expr.is_Pow:
        base = _eval_sympy_symbr(expr.args[0], var_values, t, scalar_params, local_funcs, model_preds)
        exp = _eval_sympy_symbr(expr.args[1], var_values, t, scalar_params, local_funcs, model_preds)
        return float(np.power(base, exp))

    raise NotImplementedError(f"Sympy node type not implemented: {type(expr)}")


def simulate_SymbR(equations: List[str], params: Dict[str, Any]):
    """Simulate ODEs using SymbR-trained models.

    equations: list of strings like 'x1_dot = x2*exp(a3-2)'
    params: dict described at top of file

    Returns: sol, derivatives_array
    """
    models = params.get('models', None)
    if models is None:
        raise ValueError("params must include 'models' returned by train_SymbR")

    local_funcs = params.get('local_funcs', {}) or {}
    t_span = params.get('t_span', None)
    y0 = params.get('y0', None)
    t_eval = params.get('t_eval', None)
    method = params.get('method', 'LSODA')
    atol = params.get('atol', 1e-8)
    rtol = params.get('rtol', 1e-6)
    check_nan = params.get('check_nan', True)

    scalar_params = params.get('scalar_params', params.get('obtained_coefs', {})) or {}
    # convert possible torch tensors to floats
    try:
        import torch
        scalar_params = {k: (float(v.detach().cpu().item()) if (isinstance(v, torch.nn.Parameter) or isinstance(v, torch.Tensor)) else float(v))
                         for k, v in scalar_params.items()}
    except Exception:
        scalar_params = {k: float(v) for k, v in scalar_params.items()} if scalar_params else {}

    if t_span is None or y0 is None:
        raise ValueError("params must include 't_span' and 'y0' for SymbR simulation")

    # parse equations
    parsed = [_parse_equation(eq) for eq in equations]
    lhs = [p[0] for p in parsed]
    rhs_exprs = [p[1] for p in parsed]

    # state var inference: x1_dot -> x1, else name as-is
    state_vars = []
    for name in lhs:
        if name.endswith('_dot'):
            state_vars.append(name[:-4])
        else:
            state_vars.append(name)

    if len(state_vars) != len(y0):
        raise ValueError(f"len(state_vars)={len(state_vars)} doesn't match len(y0)={len(y0)}")

    # prepare model prediction callables
    model_preds = {}
    for fname, m in models.items():
        model_preds[fname] = _make_model_pred(m)

    # RHS for integrator
    def rhs(t, y):
        var_values = {state_vars[i]: float(y[i]) for i in range(len(state_vars))}
        dydt = np.zeros_like(y, dtype=float)
        for i, expr in enumerate(rhs_exprs):
            val = _eval_sympy_symbr(expr, var_values, float(t), scalar_params, local_funcs, model_preds)
            dydt[i] = float(val)
        return dydt

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method=method, atol=atol, rtol=rtol)

    if check_nan:
        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise RuntimeError("SymbR simulation produced NaN or Inf in solution.")

    # compute derivatives array at each saved time step
    derivatives = []
    for i in range(len(sol.t)):
        tval = sol.t[i]
        yvals = sol.y[:, i]
        dydt_vals = rhs(tval, yvals)
        derivatives.append(dydt_vals)

    derivatives_array = np.array(derivatives).T

    return sol, derivatives_array


# If run as script, provide a tiny usage hint (not executed on import)
if __name__ == '__main__':
    print('This module provides simulate_SymbR(equations, params). Import and call from your script.')

