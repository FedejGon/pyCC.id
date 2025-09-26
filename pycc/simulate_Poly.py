# my_library/simulate_Poly.py  (instrumented)
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import re
from typing import Dict, Callable, List, Any, Tuple, Optional

_a_re = re.compile(r'\ba\d+\b')

def _to_float_params(params: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not params:
        return {}
    out = {}
    for k, v in params.items():
        try:
            out[k] = float(v)
        except Exception:
            raise ValueError(f"Cannot convert scalar param '{k}' value {v!r} to float.")
    return out

def _parse_equation(eq: str) -> Tuple[str, sp.Expr]:
    s = eq.replace("^", "**")
    if "=" not in s:
        raise ValueError("Equation must contain '=': " + eq)
    lhs, rhs = s.split("=", 1)
    return lhs.strip(), sp.sympify(rhs, evaluate=False)

def _eval_polynomial_model(model: Dict[str, Any], x_val: float) -> float:
    coeffs = np.asarray(model.get('coeffs', []), dtype=float)
    if coeffs.size == 0:
        return 0.0
    A0 = float(model.get('A0', 0.0))
    A1 = float(model.get('A1', 1.0))
    if A1 == 0.0:
        A1 = 1.0
    z = (float(x_val) - A0) / A1
    # coeffs expected in power order [a0, a1, a2, ...]
    return float(np.polyval(coeffs[::-1], z))

def _collect_needed_symbols(exprs: List[sp.Expr]) -> set:
    names = set()
    for e in exprs:
        for s in e.free_symbols:
            names.add(str(s))
    return names

def _eval_sympy_node(expr: sp.Basic,
                     var_values: Dict[str, float],
                     t: float,
                     models: Dict[str, Dict[str, Any]],
                     scalar_params: Dict[str, float],
                     local_funcs: Dict[str, Callable],
                     debug: bool = False) -> Any:
    """
    Evaluate sympy node. If debug True, returns tuple (value, report_str)
    where report_str is a short description for printing.
    """
    # helper to wrap results for debug
    def wrap(val, desc):
        return (float(val), desc) if debug else float(val)

    if expr.is_Number:
        return wrap(float(expr), str(expr))

    if expr.is_Symbol:
        name = str(expr)
        if name in var_values:
            return wrap(var_values[name], f"var:{name}={var_values[name]}")
        if name in scalar_params:
            return wrap(scalar_params[name], f"scalar:{name}={scalar_params[name]}")
        if name in local_funcs:
            # bare local function -> call with t
            try:
                v = float(local_funcs[name](t))
            except Exception as e:
                raise RuntimeError(f"Error calling local_funcs['{name}'](t={t}): {e}")
            return wrap(v, f"local_func:{name}(t)={v}")
        raise ValueError(f"Unknown symbol '{name}' at t={t}. Provide it in params['obtained_coefs'] or local_funcs.")

    if expr.is_Function:
        fname = expr.func.__name__
        # evaluate args with debug cascades
        if debug:
            arg_results = [ _eval_sympy_node(a, var_values, t, models, scalar_params, local_funcs, debug) for a in expr.args ]
            # arg_results are (val,desc)
            arg_vals = [r[0] for r in arg_results]
            arg_desc = ", ".join(r[1] for r in arg_results)
        else:
            arg_vals = [ _eval_sympy_node(a, var_values, t, models, scalar_params, local_funcs, debug) for a in expr.args ]

        if fname in models:
            if len(arg_vals) == 0:
                raise ValueError(f"Polynomial model '{fname}' called without args at t={t}")
            xv = arg_vals[0] if not debug else arg_vals[0][0]
            val = _eval_polynomial_model(models[fname], xv)
            desc = f"{fname}({arg_vals[0] if not debug else xv}) -> poly eval = {val}"
            return wrap(val, desc)

        if fname in local_funcs:
            try:
                if debug:
                    v = float(local_funcs[fname](*[a[0] for a in arg_results]))
                else:
                    v = float(local_funcs[fname](*arg_vals))
            except TypeError:
                v = float(local_funcs[fname](t))
            desc = f"local {fname}({arg_vals})={v}"
            return wrap(v, desc)

        numpy_map = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': abs
        }
        if fname in numpy_map:
            arg0 = arg_vals[0] if not debug else arg_vals[0][0]
            v = float(numpy_map[fname](arg0))
            desc = f"{fname}({arg0})={v}"
            return wrap(v, desc)

        raise ValueError(f"Unknown function '{fname}' in expression at t={t}")

    if expr.is_Add:
        parts = []
        sum_val = 0.0
        for a in expr.args:
            res = _eval_sympy_node(a, var_values, t, models, scalar_params, local_funcs, debug)
            if debug:
                v, d = res
                parts.append(d)
                sum_val += v
            else:
                sum_val += res
        if debug:
            return (sum_val, " + ".join(parts))
        return float(sum_val)

    if expr.is_Mul:
        parts = []
        prod = 1.0
        for a in expr.args:
            res = _eval_sympy_node(a, var_values, t, models, scalar_params, local_funcs, debug)
            if debug:
                v, d = res
                parts.append(d)
                prod *= v
            else:
                prod *= res
        if debug:
            return (prod, " * ".join(parts))
        return float(prod)

    if expr.is_Pow:
        base = _eval_sympy_node(expr.args[0], var_values, t, models, scalar_params, local_funcs, debug)
        exp = _eval_sympy_node(expr.args[1], var_values, t, models, scalar_params, local_funcs, debug)
        if debug:
            v = float(np.power(base[0], exp[0]))
            return (v, f"pow({base[1]}, {exp[1]})={v}")
        else:
            return float(np.power(base, exp))

    raise NotImplementedError(f"Sympy node type not implemented: {type(expr)}")

def prepare_sim_params(models: Dict[str, Any],
                       final_scalars: Dict[str, float],
                       equations: List[str],
                       t_span=(0.0, 10.0),
                       y0: Optional[List[float]] = None,
                       t_eval: Optional[np.ndarray] = None,
                       local_funcs: Optional[Dict[str, Callable]] = None,
                       method: str = "LSODA",
                       atol: float = 1e-8,
                       rtol: float = 1e-6) -> Dict[str, Any]:

    scalar_params = _to_float_params(final_scalars or {})
    used = set()
    for eq in equations:
        used.update(_a_re.findall(eq))
    missing = sorted([a for a in used if a not in scalar_params])
    if missing:
        for a in missing:
            scalar_params[a] = 0.0
        print(f"Warning: auto-filled missing scalar params {missing} with 0.0.")

    params = {
        "models": models or {},
        "obtained_coefs": scalar_params,
        "local_funcs": local_funcs or {},
        "t_span": tuple(t_span),
        "y0": list(y0) if y0 is not None else None,
        "t_eval": t_eval,
        "method": method,
        "atol": atol,
        "rtol": rtol,
        "check_nan": True,
    }
    if params["y0"] is None:
        raise ValueError("y0 must be provided")
    return params

def simulate_Poly(equations: List[str], params: Dict[str, Any]):
    # --- START OF MODIFIED PARAMETER READING (SIMPLIFIED, NO TORCH) ---
    
    # 1. Get the scalar dictionary using ONLY 'scalar_params'. Defaults to empty dict.
    scalar_input = params.get('scalar_params', {})
    
    # 2. Final conversion to a dictionary of pure floats using the existing utility function.
    # This function handles conversion errors for non-float/non-convertible values.
    # We rely on _to_float_params to handle basic types (like NumPy scalars)
    # and raise a clean ValueError if a value is not convertible.
    scalar_params = _to_float_params(scalar_input)
    
    # --- END OF MODIFIED PARAMETER READING ---

    if not isinstance(equations, (list, tuple)):
        raise ValueError("equations must be a list of strings")

    models = params.get('models', {}) or {}
    local_funcs = params.get('local_funcs', {}) or {}
    t_span = params.get('t_span', None)
    y0 = params.get('y0', None)
    t_eval = params.get('t_eval', None)
    method = params.get('method', 'LSODA')
    atol = params.get('atol', 1e-8)
    rtol = params.get('rtol', 1e-6)
    check_nan = params.get('check_nan', True)
    debug = bool(params.get('debug', False))

    if t_span is None or y0 is None:
        raise ValueError("params must include 't_span' and 'y0'")

    # ... (The rest of the function remains identical)
    parsed = [_parse_equation(eq) for eq in equations]
    lhs = [p[0] for p in parsed]
    rhs_exprs = [p[1] for p in parsed]

    state_vars = []
    for name in lhs:
        state_vars.append(name[:-4] if name.endswith('_dot') else name)

    if len(state_vars) != len(y0):
        raise ValueError(f"len(state_vars)={len(state_vars)} doesn't match len(y0)={len(y0)}")

    needed_symbols = _collect_needed_symbols(rhs_exprs)
    func_names = set(models.keys()) | set(local_funcs.keys())
    leftover = needed_symbols - set(state_vars) - func_names

    truly_missing = []
    for sym_name in leftover:
        if sym_name in scalar_params:
            continue
        try:
            float(sym_name)
            continue
        except ValueError:
            truly_missing.append(sym_name)
            scalar_params[sym_name] = 0.0

    if truly_missing:
        print(f"Warning: Missing scalar params {truly_missing} auto-filled with 0.0.")

    t0 = float(t_span[0])
    y0_arr = np.asarray(y0, dtype=float)
    if debug:
        print("=== DEBUG: single-eval check at t0,y0 ===")
        sample_var_values = {state_vars[i]: float(y0_arr[i]) for i in range(len(state_vars))}
        for i, expr in enumerate(rhs_exprs):
            try:
                res = _eval_sympy_node(expr, sample_var_values, t0, models, scalar_params, local_funcs, debug=True)
                print(f"RHS[{i}] expr: {expr}\n -> value = {res[0]:.6e}\n -> breakdown: {res[1]}")
            except Exception as e:
                print(f"  ERROR evaluating RHS[{i}] = {expr} at t0: {e}")
        print("=== END DEBUG single-eval ===")

    def rhs(t, y):
        var_values = {state_vars[i]: float(y[i]) for i in range(len(state_vars))}
        dydt = np.zeros_like(y, dtype=float)
        for i, expr in enumerate(rhs_exprs):
            val = _eval_sympy_node(expr, var_values, float(t), models, scalar_params, local_funcs, debug=False)
            dydt[i] = float(val)
        return dydt

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method=method, atol=atol, rtol=rtol)

    if check_nan and (np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y))):
        raise RuntimeError("Simulation produced NaN/Inf")

    return sol
