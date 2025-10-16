"""
simulate_from_evals_interp.py

Lightweight simulator that receives only `evals` (discrete x,y pairs for each
unknown function f_i) and integrates a system of ODEs provided as SymPy-like
strings (e.g. 'x1_dot = x2 - f1(x1)').

This module builds robust, efficient 1D interpolants for each f_i and uses
those interpolants during integration. Default interpolator: PCHIP
shape-preserving cubic (monotonic), falling back to linear interpolation.

API:
    sol, derivatives = simulate_from_evals(equations, params)

Required params:
    - 'evals': list [x_f1, y_f1, x_f2, y_f2, ...] (pairs for each f)
    - 't_span': (t0, tf)
    - 'y0': initial state vector matching equations order

Optional params:
    - 'function_names': list like ['f1','f2'] (defaults to ['f1','f2',...])
    - 'interp_method': 'pchip' (default), 'linear', 'cubic', or 'spline'
    - 'smoothing': float used only with 'spline' (UnivariateSpline)
    - 'extrapolate': bool default True (allow extrapolation)
    - 'local_funcs': dict for e.g. {'F_ext': lambda t: ...}
    - 'scalar_params' or 'obtained_coefs': dict of scalar coefficients (a1,a2,...)
    - 't_eval','method','atol','rtol','check_nan','print_models'

Design choices:
 - PCHIP is used by default because it's shape-preserving and avoids
   oscillations that cubic splines can produce on uneven/noisy data.
 - If SciPy's PchipInterpolator is unavailable, we automatically fall back to
   scipy.interpolate.interp1d('linear') or numpy.interp.

"""

from typing import List, Dict, Any, Callable, Tuple
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import warnings

# preferred interpolators
try:
    from scipy.interpolate import PchipInterpolator, interp1d, UnivariateSpline
except Exception:
    PchipInterpolator = None
    interp1d = None
    UnivariateSpline = None


def _parse_equation(eq: str) -> Tuple[str, sp.Expr]:
    s = eq.replace('^', '**')
    if '=' not in s:
        raise ValueError("Equation must contain '='")
    lhs, rhs = s.split('=', 1)
    return lhs.strip(), sp.sympify(rhs, evaluate=False)


def _make_interpolator(xp: np.ndarray, yp: np.ndarray, method: str = 'pchip',
                       smoothing: float = 0.0, extrapolate: bool = True):
    """Return a function pred(xq)->yq using selected interpolation method.

    Methods:
      - 'pchip' : PCHIP interpolator (shape-preserving cubic)
      - 'linear' : linear interp
      - 'cubic' : cubic spline via interp1d (may oscillate)
      - 'spline' : UnivariateSpline with smoothing
    """
    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if xp.ndim != 1 or yp.ndim != 1 or xp.size != yp.size:
        raise ValueError('xp and yp must be 1D arrays of same length')
    # sort
    idx = np.argsort(xp)
    xp = xp[idx]
    yp = yp[idx]

    # If not enough points for higher-order methods, fallback to linear
    if xp.size < 2:
        # constant function
        const = float(yp.ravel()[0]) if xp.size == 1 else 0.0
        return lambda xq, c=const: np.full_like(np.asarray(xq, dtype=float), c, dtype=float)

    # PCHIP preferred
    if method == 'pchip' and PchipInterpolator is not None:
        try:
            interp = PchipInterpolator(xp, yp, extrapolate=extrapolate)
            return lambda xq, interp=interp: np.asarray(interp(np.asarray(xq, dtype=float))).ravel().astype(float)
        except Exception:
            warnings.warn('PCHIP construction failed; falling back to linear')
            method = 'linear'

    # UnivariateSpline
    if method == 'spline' and UnivariateSpline is not None:
        try:
            us = UnivariateSpline(xp, yp, s=float(smoothing))
            # control extrapolation: UnivariateSpline extrapolates by default
            if not extrapolate:
                def pred_nox(xq, us=us, xp=xp):
                    xq = np.asarray(xq, dtype=float)
                    yq = us(xq)
                    left = yp[0]
                    right = yp[-1]
                    yq[xq < xp[0]] = left
                    yq[xq > xp[-1]] = right
                    return np.asarray(yq).ravel().astype(float)
                return pred_nox
            return lambda xq, us=us: np.asarray(us(np.asarray(xq, dtype=float))).ravel().astype(float)
        except Exception:
            warnings.warn('UnivariateSpline failed; falling back to linear')
            method = 'linear'

    # interp1d based methods (linear/cubic)
    if interp1d is not None and method in ('linear', 'cubic'):
        kind = method
        try:
            interp = interp1d(xp, yp, kind=kind, bounds_error=False, fill_value=(yp[0], yp[-1]) if not extrapolate else None)
            return lambda xq, interp=interp: np.asarray(interp(np.asarray(xq, dtype=float))).ravel().astype(float)
        except Exception:
            warnings.warn('interp1d failed; falling back to numpy.interp')

    # final fallback: numpy.interp (linear, no extrapolate beyond endpoints -> edge values)
    def numpy_interp_pred(xq, xp=xp, yp=yp):
        xq = np.asarray(xq, dtype=float)
        yq = np.interp(xq, xp, yp, left=yp[0], right=yp[-1])
        return yq.astype(float)

    return numpy_interp_pred


def _eval_sympy_using_interpolants(expr: sp.Basic,
                                   var_values: Dict[str, float],
                                   t: float,
                                   scalar_params: Dict[str, float],
                                   local_funcs: Dict[str, Callable],
                                   model_preds: Dict[str, Callable]):
    """Evaluate sympy expression where f_i(...) calls model_preds[f_i].

    Supports basic arithmetic and common unary numpy functions.
    """
    # numbers
    if expr.is_Number:
        return float(expr)
    if expr.is_Symbol:
        name = str(expr)
        if name in var_values:
            return float(var_values[name])
        if name in scalar_params:
            return float(scalar_params[name])
        if name in local_funcs:
            return float(local_funcs[name](t))
        raise ValueError(f"Unknown symbol '{name}' at t={t}")

    if expr.is_Function:
        fname = expr.func.__name__
        arg_vals = [_eval_sympy_using_interpolants(a, var_values, t, scalar_params, local_funcs, model_preds) for a in expr.args]
        if fname in model_preds:
            if len(arg_vals) != 1:
                raise ValueError(f"Model {fname} expected 1 arg, got {len(arg_vals)}")
            xq = np.asarray([arg_vals[0]], dtype=float)
            y = model_preds[fname](xq)
            return float(np.asarray(y).ravel()[0])
        if fname in local_funcs:
            try:
                return float(local_funcs[fname](*arg_vals))
            except TypeError:
                return float(local_funcs[fname](t))
        numpy_map = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': abs
        }
        if fname in numpy_map:
            return float(numpy_map[fname](arg_vals[0]))
        raise ValueError(f"Unknown function '{fname}' in expression")

    if expr.is_Add:
        return float(sum(_eval_sympy_using_interpolants(a, var_values, t, scalar_params, local_funcs, model_preds) for a in expr.args))
    if expr.is_Mul:
        prod = 1.0
        for a in expr.args:
            prod *= _eval_sympy_using_interpolants(a, var_values, t, scalar_params, local_funcs, model_preds)
        return float(prod)
    if expr.is_Pow:
        base = _eval_sympy_using_interpolants(expr.args[0], var_values, t, scalar_params, local_funcs, model_preds)
        exp = _eval_sympy_using_interpolants(expr.args[1], var_values, t, scalar_params, local_funcs, model_preds)
        return float(np.power(base, exp))

    raise NotImplementedError(f"Sympy node type not implemented: {type(expr)}")


def simulate_from_evals_interp(equations: List[str], params: Dict[str, Any]):
    """Simulate ODEs using only discrete evaluations (evals).

    params required keys:
      - 'evals' : list [x1,y1,x2,y2,...]
      - 't_span'
      - 'y0'

    Optional keys: see top docstring.
    Returns: sol, derivatives_array
    """
    evals = params.get('evals', None)
    if evals is None:
        raise ValueError("params must include 'evals' list of pairs [x1,y1,...]")
    function_names = params.get('function_names', None)
    interp_method = params.get('interp_method', 'pchip')
    smoothing = float(params.get('smoothing', 0.0))
    extrapolate = bool(params.get('extrapolate', True))

    local_funcs = params.get('local_funcs', {}) or {}
    t_span = params.get('t_span', None)
    y0 = params.get('y0', None)
    t_eval = params.get('t_eval', None)
    method = params.get('method', 'LSODA')
    atol = params.get('atol', 1e-8)
    rtol = params.get('rtol', 1e-6)
    check_nan = params.get('check_nan', True)

    scalar_params = params.get('scalar_params', params.get('obtained_coefs', {})) or {}
    try:
        import torch
        scalar_params = {k: (float(v.detach().cpu().item()) if (isinstance(v, torch.nn.Parameter) or isinstance(v, torch.Tensor)) else float(v))
                         for k, v in scalar_params.items()}
    except Exception:
        scalar_params = {k: float(v) for k, v in scalar_params.items()} if scalar_params else {}

    if t_span is None or y0 is None:
        raise ValueError("params must include 't_span' and 'y0'")

    parsed = [_parse_equation(eq) for eq in equations]
    lhs = [p[0] for p in parsed]
    rhs_exprs = [p[1] for p in parsed]
    state_vars = [name[:-4] if name.endswith('_dot') else name for name in lhs]
    if len(state_vars) != len(y0):
        raise ValueError(f"len(state_vars)={len(state_vars)} doesn't match len(y0)={len(y0)}")

    # Build interpolants from evals
    if len(evals) % 2 != 0:
        raise ValueError("params['evals'] must be pairs [x1,y1,x2,y2,...]")
    nfuncs = len(evals) // 2
    if function_names is None:
        function_names = [f'f{i+1}' for i in range(nfuncs)]
    model_preds: Dict[str, Callable] = {}
    for i in range(nfuncs):
        xp = np.asarray(evals[2*i], dtype=float)
        yp = np.asarray(evals[2*i+1], dtype=float)
        fname = function_names[i] if i < len(function_names) else f'f{i+1}'
        model_preds[fname] = _make_interpolator(xp, yp, method=interp_method, smoothing=smoothing, extrapolate=extrapolate)

    # RHS
    def rhs(t, y):
        var_map = {state_vars[i]: float(y[i]) for i in range(len(state_vars))}
        dydt = np.zeros_like(y, dtype=float)
        for i, expr in enumerate(rhs_exprs):
            val = _eval_sympy_using_interpolants(expr, var_map, float(t), scalar_params, local_funcs, model_preds)
            dydt[i] = float(val)
        return dydt

    # print info
    if bool(params.get('print_models', True)):
        print('=== Interpolants used in simulation ===')
        for fname in sorted(model_preds.keys()):
            print(f"{fname}: interp_method={interp_method}")
        print('=== end interpolants ===')

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method=method, atol=atol, rtol=rtol)

    if check_nan and (np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y))):
        raise RuntimeError('Simulation produced NaN or Inf in solution')

    derivatives = [rhs(sol.t[i], sol.y[:, i]) for i in range(len(sol.t))]
    derivatives_array = np.array(derivatives).T
    return sol, derivatives_array


if __name__ == '__main__':
    print('This module provides simulate_from_evals(equations, params).')

