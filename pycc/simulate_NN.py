# my_library/simulate.py
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import torch
from typing import Dict, Callable, List, Any, Tuple, Optional

# --------------------------
# Helpers
# --------------------------
def _to_float_params(params: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Convert scalar params (torch.nn.Parameter, tensor or float) to float dict."""
    if not params:
        return {}
    out = {}
    for k, v in params.items():
        try:
            if isinstance(v, torch.nn.Parameter) or isinstance(v, torch.Tensor):
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = float(v)
        except Exception:
            out[k] = float(v)
    return out

def _parse_equation(eq: str) -> Tuple[str, sp.Expr]:
    """Parse 'lhs = rhs' into lhs string and sympy expression for rhs."""
    s = eq.replace("^", "**")
    lhs, rhs = s.split("=")
    lhs_name = lhs.strip()
    rhs_expr = sp.sympify(rhs, evaluate=False)  # keep function calls as Function nodes
    return lhs_name, rhs_expr

def _eval_sympy_node(expr: sp.Basic,
                     var_values: Dict[str, float],
                     t: float,
                     models: Dict[str, torch.nn.Module],
                     scalar_params: Dict[str, float],
                     local_funcs: Dict[str, Callable]) -> float:
    """
    Recursively evaluate a sympy node to float.
    - models: dict with learned NNs (e.g. 'f1'->NNModel)
    - local_funcs: dict of user functions (e.g. 'F_ext': lambda t: ...)
    - scalar_params: floats for a1,a2...
    """
    # Numbers
    if expr.is_Number:
        return float(expr)

    # Symbol -> state var, scalar param, or bare local function name
    if expr.is_Symbol:
        name = str(expr)
        if name in var_values:
            return float(var_values[name])
        if name in scalar_params:
            return float(scalar_params[name])
        if name in local_funcs:
            # bare symbol like F_ext -> treat as time-dependent, call F_ext(t)
            return float(local_funcs[name](t))
        raise ValueError(f"Unknown symbol '{name}' at t={t}")

    # Function call: f1(x), F_ext(t), sin(x), etc.
    if expr.is_Function:
        fname = expr.func.__name__
        # evaluate arguments
        arg_vals = [_eval_sympy_node(a, var_values, t, models, scalar_params, local_funcs) for a in expr.args]

        # learned NN function (f1, f2, ...)
        if fname in models:
            model = models[fname]
            model.eval()
            with torch.no_grad():
                xt = torch.tensor([[float(arg_vals[0])]], dtype=torch.float32)
                y = model(xt).cpu().numpy().ravel()[0]
            return float(y)

        # user-supplied local function (call with args if possible)
        if fname in local_funcs:
            try:
                return float(local_funcs[fname](*arg_vals))
            except TypeError:
                # fallback: maybe local_funcs expects only t
                return float(local_funcs[fname](t))

        # math functions -> numpy
        numpy_map = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': abs
        }
        if fname in numpy_map:
            return float(numpy_map[fname](arg_vals[0]))

        raise ValueError(f"Unknown function '{fname}' in expression")

    # Add / Mul / Pow
    if expr.is_Add:
        return float(sum(_eval_sympy_node(a, var_values, t, models, scalar_params, local_funcs) for a in expr.args))
    if expr.is_Mul:
        prod = 1.0
        for a in expr.args:
            prod *= _eval_sympy_node(a, var_values, t, models, scalar_params, local_funcs)
        return float(prod)
    if expr.is_Pow:
        base = _eval_sympy_node(expr.args[0], var_values, t, models, scalar_params, local_funcs)
        exp = _eval_sympy_node(expr.args[1], var_values, t, models, scalar_params, local_funcs)
        return float(np.power(base, exp))

    raise NotImplementedError(f"Sympy node type not implemented: {type(expr)}")

# --------------------------
# NN-based simulator
# --------------------------
def simulate_NN(equations, params):
    """
    Simulate using trained NN components.
    Required keys inside params:
      - 'models' : dict of NN models returned by training (e.g. {'f1': NNModel, 'f2': NNModel})
      - 'obtained_coefs' or 'scalar_params' : dict of scalar params (torch.nn.Parameter or floats)
      - 'local_funcs' : dict mapping name->callable (e.g. {'F_ext': lambda t: F0*np.cos(Omega*t)})
      - 't_span' : (t0, tf)
      - 'y0' : initial state list/array, ordered to match equations
      - 't_eval' : array-like times (optional)
      Optional keys:
      - 'method' : integrator name for solve_ivp (defaults to 'LSODA')
      - 'atol', 'rtol', 'check_nan' : numeric/boolean
    """
    # Extract and validate params
    models = params.get('models', {})
    if models is None:
        models = {}
    obtained_coefs = params.get('obtained_coefs', params.get('scalar_params', {}))
    local_funcs = params.get('local_funcs', {}) or {}
    t_span = params.get('t_span', None)
    y0 = params.get('y0', None)
    t_eval = params.get('t_eval', None)
    method = params.get('method', 'LSODA')
    atol = params.get('atol', 1e-8)
    rtol = params.get('rtol', 1e-6)
    check_nan = params.get('check_nan', True)

    if t_span is None or y0 is None:
        raise ValueError("params must include 't_span' and 'y0' for NN simulation")

    # move models to cpu and eval mode
    for m in models.values():
        m.eval()
        m.to('cpu')

    scalar_params = _to_float_params(obtained_coefs or {})

    # parse equations
    parsed = [_parse_equation(eq) for eq in equations]
    lhs = [p[0] for p in parsed]
    rhs_exprs = [p[1] for p in parsed]

    # infer state var names from lhs 'x1_dot' -> 'x1'
    state_vars = []
    for name in lhs:
        if name.endswith('_dot'):
            state_vars.append(name[:-4])
        else:
            state_vars.append(name)

    if len(state_vars) != len(y0):
        raise ValueError(f"len(state_vars)={len(state_vars)} doesn't match len(y0)={len(y0)}")

    # RHS to pass to solve_ivp
    def rhs(t, y):
        var_values = {state_vars[i]: float(y[i]) for i in range(len(state_vars))}
        dydt = np.zeros_like(y, dtype=float)
        for i, expr in enumerate(rhs_exprs):
            val = _eval_sympy_node(expr, var_values, float(t), models, scalar_params, local_funcs)
            dydt[i] = float(val)
        return dydt

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method=method, atol=atol, rtol=rtol)

    if check_nan:
        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise RuntimeError("Simulation produced NaN or Inf values in solution.")

    return sol

