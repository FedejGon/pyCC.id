# add into my_library/simulate.py (or a similar file)
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from typing import Dict, Callable, List, Any, Tuple, Optional

def _parse_equation(eq: str) -> Tuple[str, sp.Expr]:
    s = eq.replace("^", "**")
    lhs, rhs = s.split("=")
    lhs_name = lhs.strip()
    rhs_expr = sp.sympify(rhs, evaluate=False)   # keep function calls as Function nodes
    return lhs_name, rhs_expr

def _eval_sympy_theory(expr: sp.Basic,
                       var_values: Dict[str, float],
                       t: float,
                       scalar_params: Dict[str, float],
                       local_funcs: Dict[str, Callable]) -> float:
    """
    Evaluate a sympy expression node to a float using:
      - var_values: mapping variable name -> float (state values)
      - scalar_params: dict of floats a1,a2,...
      - local_funcs: mapping name->callable. For f1(x) -> call local_funcs['f1'](x_val),
                     for F_ext(t) -> call local_funcs['F_ext'](t). Fallbacks attempted if call fails.
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
            # treat bare symbol as time-dependent function
            return float(local_funcs[name](t))
        raise ValueError(f"Unknown symbol '{name}' at t={t}")

    # Function call: f1(x), F_ext(t), sin(x), ...
    if expr.is_Function:
        fname = expr.func.__name__
        # evaluate args recursively
        arg_vals = [_eval_sympy_theory(a, var_values, t, scalar_params, local_funcs) for a in expr.args]

        # user-supplied function e.g. f1, f2, F_ext ...
        if fname in local_funcs:
            func = local_funcs[fname]
            # try calling with evaluated arguments first
            try:
                return float(func(*arg_vals))
            except TypeError:
                # fallback: try calling with t only
                try:
                    return float(func(t))
                except Exception as e:
                    raise RuntimeError(f"local_funcs['{fname}'] call failed for args {arg_vals} and t={t}: {e}")
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
        return float(sum(_eval_sympy_theory(a, var_values, t, scalar_params, local_funcs) for a in expr.args))
    # Mul
    if expr.is_Mul:
        prod = 1.0
        for a in expr.args:
            prod *= _eval_sympy_theory(a, var_values, t, scalar_params, local_funcs)
        return float(prod)
    # Pow
    if expr.is_Pow:
        base = _eval_sympy_theory(expr.args[0], var_values, t, scalar_params, local_funcs)
        exp = _eval_sympy_theory(expr.args[1], var_values, t, scalar_params, local_funcs)
        return float(np.power(base, exp))

    raise NotImplementedError(f"Sympy node type not implemented: {type(expr)}")


def simulate_th(equations: List[str], params: Dict[str, Any]):
    """
    Generic theoretical simulator driven by equation strings.

    Required in params:
      - 'local_funcs': dict mapping function names used in equations -> callables (e.g. {'f1': f1, 'f2': f2, 'F_ext': F_ext})
      - 't_span': (t0, tf)
      - 'y0': initial state vector (ordered to match equations)
    Optional:
      - 't_eval', 'method', 'atol', 'rtol', 'scalar_params' (dict floats or torch params), 'check_nan'
    """

    local_funcs = params.get('local_funcs', {}) or {}
    t_span = params.get('t_span', None)
    y0 = params.get('y0', None)
    t_eval = params.get('t_eval', None)
    method = params.get('method', 'LSODA')
    atol = params.get('atol', 1e-8)
    rtol = params.get('rtol', 1e-6)
    scalar_params = params.get('scalar_params', params.get('obtained_coefs', {})) or {}
    # if scalar_params may contain torch tensors/parameters, convert to floats
    try:
        import torch
        scalar_params = {k: (float(v.detach().cpu().item()) if isinstance(v, torch.nn.Parameter) or isinstance(v, torch.Tensor) else float(v))
                         for k, v in scalar_params.items()}
    except Exception:
        scalar_params = {k: float(v) for k, v in scalar_params.items()} if scalar_params else {}

    check_nan = params.get('check_nan', True)

    if t_span is None or y0 is None:
        raise ValueError("params must include 't_span' and 'y0' for theoretical simulation")

    # parse equations
    parsed = [_parse_equation(eq) for eq in equations]
    lhs = [p[0] for p in parsed]
    rhs_exprs = [p[1] for p in parsed]

    # infer state vars from lhs (x1_dot -> x1)
    state_vars = []
    for name in lhs:
        if name.endswith('_dot'):
            state_vars.append(name[:-4])
        else:
            state_vars.append(name)

    if len(state_vars) != len(y0):
        raise ValueError(f"len(state_vars)={len(state_vars)} doesn't match len(y0)={len(y0)}")

    # RHS function for integrator
    def rhs(t, y):
        var_values = {state_vars[i]: float(y[i]) for i in range(len(state_vars))}
        dydt = np.zeros_like(y, dtype=float)
        for i, expr in enumerate(rhs_exprs):
            val = _eval_sympy_theory(expr, var_values, float(t), scalar_params, local_funcs)
            dydt[i] = float(val)
        return dydt

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method=method, atol=atol, rtol=rtol)

    if check_nan:
        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise RuntimeError("Theoretical simulation produced NaN or Inf in solution.")


    # --- Part of the code to calculate and return derivatives ---
    # Create an empty list to store the derivative arrays
    derivatives = []
    # Loop through each time step in the solution
    for i in range(len(sol.t)):
        # Extract the state values (y) for the current time step (t)
        t_val = sol.t[i]
        y_vals = sol.y[:, i]
        # Re-evaluate the rhs function to get the derivatives
        dydt_vals = rhs(t_val, y_vals)
        # Append the calculated derivatives to the list
        derivatives.append(dydt_vals)
    
    # Convert the list of derivative arrays into a single NumPy array
    derivatives_array = np.array(derivatives).T

    return sol, derivatives_array

