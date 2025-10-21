"""
simulate_SymbR.py

Simulator compatible with train_SymbR output and SR postprocessing. Updated
printing to display symbolic expressions when available (from sr_results or
from PySR model objects).

Also supports fitting PySR models from discrete evals; defaults for PySR are
set to safer operators and default niterations/populations as requested.
"""

import warnings
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

# Optional import of PySR (only used when fitting from evals)
try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None

def _simplify_and_format_expr(expr_str: str, precision: int = 5) -> str:
    """
    Uses sympy to expand and simplify a symbolic expression string for better readability.
    - Expands products (e.g., (x+1)*(x-2) -> x**2 - x - 2)
    - Rounds floating-point numbers to a specified precision.
    """
    try:
        # Sympy needs to know what the variables are. 'x0' is the standard for single-variable PySR.
        x0 = sp.Symbol('x0')
        
        # Convert the string into a manipulatable sympy expression
        expr = sp.sympify(expr_str, locals={'x0': x0})
        
        # Expand the expression to resolve products of sums
        # This turns (a-x)*(b*x**2 - c) into a standard polynomial form
        expr = sp.expand(expr)
        
        # This part recursively finds all numbers in the expression and rounds them
        simplified_expr = expr.xreplace({
            n: sp.Float(round(n, precision)) for n in expr.atoms(sp.Number)
        })

        # Convert the simplified expression back to a human-readable string
        return str(simplified_expr)
    except (sp.SympifyError, TypeError, Exception):
        # If simplification fails for any reason, just return the original cleaned string
        return " ".join(str(expr_str).split())

def _parse_equation(eq: str) -> Tuple[str, sp.Expr]:
    s = eq.replace("^", "**")
    if '=' not in s:
        raise ValueError("Equation must contain '='")
    lhs, rhs = s.split('=', 1)
    lhs_name = lhs.strip()
    rhs_expr = sp.sympify(rhs, evaluate=False)
    return lhs_name, rhs_expr


def _attach_attr(fn: Callable, **attrs):
    for k, v in attrs.items():
        try:
            setattr(fn, k, v)
        except Exception:
            pass
    return fn


def _make_model_pred(model_obj: Any):
    if model_obj is None:
        fn = lambda x: np.zeros_like(np.asarray(x, dtype=float), dtype=float)
        return _attach_attr(fn, _source='zero')

    if callable(model_obj):
        def wrap_callable(xq, fn=model_obj):
            xq = np.asarray(xq, dtype=float)
            try:
                yq = fn(xq)
            except Exception:
                yq = np.array([fn(float(xi)) for xi in xq])
            return np.asarray(yq).ravel().astype(float)
        return _attach_attr(wrap_callable, _source='callable')

    if isinstance(model_obj, dict):
        if ('pysr_model' in model_obj) or ('const' in model_obj):
            if model_obj.get('pysr_model') is None:
                const_val = float(model_obj.get('const', 0.0))
                fn = lambda x, c=const_val: np.full_like(np.asarray(x, dtype=float), c, dtype=float)
                return _attach_attr(fn, _source='pysr_const', _const=const_val, _expr=model_obj.get('expr'))
            pm = model_obj.get('pysr_model')
            A0 = float(model_obj.get('A0', 0.0))
            A1 = float(model_obj.get('A1', 1.0))
            def pred_pysr(xq, pm=pm, A0=A0, A1=A1):
                xq = np.asarray(xq, dtype=float)
                z = (xq - A0) / A1
                Xz = z.reshape(-1, 1)
                try:
                    yhat = pm.predict(Xz)
                except Exception as e:
                    warnings.warn(f"PySR predict failed: {e}")
                    yhat = np.zeros(len(Xz))
                return np.asarray(yhat).ravel().astype(float)
            return _attach_attr(pred_pysr, _source='pysr', _A0=A0, _A1=A1, _pymodel=pm, _expr=model_obj.get('expr'))

        if 'func' in model_obj and callable(model_obj['func']):
            return _make_model_pred(model_obj['func'])

        xarr = None
        yarr = None
        if 'x' in model_obj and 'y' in model_obj:
            xarr = np.asarray(model_obj['x'], dtype=float)
            yarr = np.asarray(model_obj['y'], dtype=float)
        elif 'x_plot' in model_obj and 'y_plot' in model_obj:
            xarr = np.asarray(model_obj['x_plot'], dtype=float)
            yarr = np.asarray(model_obj['y_plot'], dtype=float)
        if xarr is not None and yarr is not None:
            idx = np.argsort(xarr)
            xp = xarr[idx]
            yp = yarr[idx]
            def interp_fn(xq, xp=xp, yp=yp):
                xq = np.asarray(xq, dtype=float)
                yq = np.interp(xq, xp, yp, left=yp[0], right=yp[-1])
                return yq.astype(float)
            return _attach_attr(interp_fn, _source='interp', _xp=xp, _yp=yp)

    raise ValueError('Unsupported model object passed to _make_model_pred')


def _extract_expr_from_pysr(pm) -> Optional[str]:
    """
    Precisely extracts the single best symbolic equation string from a PySR model.
    """
    if pm is None:
        return None
    
    try:
        # The best way to get the final equation is using get_best()
        if hasattr(pm, 'get_best'):
            # This returns a pandas Series representing the best equation
            best_equation_series = pm.get_best()
            
            # The actual formula is in the 'equation' column of that Series
            if 'equation' in best_equation_series:
                return str(best_equation_series['equation'])
                
        # Fallback for different PySR versions or model states
        if hasattr(pm, 'equations_'):
            # Find the equation with the best score (lowest is best in PySR)
            best_row = pm.equations_.iloc[pm.equations_['score'].idxmin()]
            if 'equation' in best_row:
                return str(best_row['equation'])
                
        warnings.warn("Could not find a 'best' equation. Falling back to string representation.")
        return str(pm) # Last resort
        
    except Exception as e:
        warnings.warn(f"An error occurred while extracting the PySR equation: {e}")
        return None



def _fit_evals_with_pysr(evals: List[Any], function_names: Optional[List[str]] = None,
                         sr_params: Optional[Dict[str, Any]] = None):
    """Fit PySR models from evals = [x1,y1, x2,y2, ...].

    Defaults requested by user:
        niterations = 200, populations = 15
        unary_operators = ["cos","sin","exp","log","sqrt","tanh"]
        binary_operators = ["+","-","*"]
        model_selection = "best"
        loss = "loss(x, y) = (x - y)^2"

    The user may pass sr_params={'niterations':..,'populations':.., ...} to override.
    """
    if function_names is None:
        n = len(evals) // 2
        function_names = [f'f{i+1}' for i in range(n)]
    if len(evals) % 2 != 0:
        raise ValueError('evals must contain pairs [x1,y1,x2,y2,...]')

    models_out = {}
    nfuncs = len(evals) // 2
    for i in range(nfuncs):
        xp = np.asarray(evals[2*i], dtype=float)
        yp = np.asarray(evals[2*i+1], dtype=float)
        fname = function_names[i] if i < len(function_names) else f'f{i+1}'

        A0 = float((xp.max() + xp.min())/2.0)
        A1 = float((xp.max() - xp.min())/2.0)
        if A1 == 0.0:
            A1 = 1.0

        if PySRRegressor is not None:
            # prepare PySR kwargs with requested defaults and allow overrides
            user_params = dict(sr_params) if sr_params else {}
            niterations = int(user_params.pop('niterations', 200))
            populations = int(user_params.pop('populations', 15))
            # base kwargs
            base_kwargs = {
                'niterations': niterations,
                'populations': populations,
                'binary_operators': ["+", "-", "*"],
                'unary_operators': ["cos", "sin", "exp", "log", "sqrt", "tanh"],
                'model_selection': "best",
                'loss': "loss(x, y) = (x - y)^2",
                'verbosity': 0,
            }
            # merge user params (remaining keys) overriding defaults
            pysr_kwargs = {**base_kwargs, **user_params}
            try:
                model = PySRRegressor(**pysr_kwargs)
                X = ((xp - A0) / A1).reshape(-1, 1)
                model.fit(X, yp)
                expr = _extract_expr_from_pysr(model)
                models_out[fname] = {'pysr_model': model, 'A0': A0, 'A1': A1, 'expr': expr}
                continue
            except Exception as e:
                warnings.warn(f'PySR fitting for {fname} failed, falling back to interp: {e}')

        models_out[fname] = {'x': xp, 'y': yp}

    return models_out


def _eval_sympy_symbr(expr: sp.Basic,
                      var_values: Dict[str, float],
                      t: float,
                      scalar_params: Dict[str, float],
                      local_funcs: Dict[str, Callable],
                      model_preds: Dict[str, Callable]):
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
        args = [ _eval_sympy_symbr(a, var_values, t, scalar_params, local_funcs, model_preds)
                 for a in expr.args ]
        if fname in model_preds:
            if len(args) != 1:
                raise ValueError(f"Model {fname} expected 1 arg, got {len(args)}")
            xval = np.asarray([args[0]], dtype=float)
            y = model_preds[fname](xval)
            return float(np.asarray(y).ravel()[0])
        if fname in local_funcs:
            try:
                return float(local_funcs[fname](*args))
            except TypeError:
                return float(local_funcs[fname](t))
        numpy_map = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': abs
        }
        if fname in numpy_map:
            return float(numpy_map[fname](args[0]))
        raise ValueError(f"Unknown function '{fname}' in expression")

    if expr.is_Add:
        return float(sum(_eval_sympy_symbr(a, var_values, t, scalar_params, local_funcs, model_preds) for a in expr.args))
    if expr.is_Mul:
        prod = 1.0
        for a in expr.args:
            prod *= _eval_sympy_symbr(a, var_values, t, scalar_params, local_funcs, model_preds)
        return float(prod)
    if expr.is_Pow:
        base = _eval_sympy_symbr(expr.args[0], var_values, t, scalar_params, local_funcs, model_preds)
        exp = _eval_sympy_symbr(expr.args[1], var_values, t, scalar_params, local_funcs, model_preds)
        return float(np.power(base, exp))

    raise NotImplementedError(f"Sympy node type not implemented: {type(expr)}")


def simulate_SymbR(equations: List[str], params: Dict[str, Any]):
    models = params.get('models', None)
    evals = params.get('evals', None)
    function_names = params.get('function_names', None)
    sr_params = params.get('pysr', None)

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

    model_preds: Dict[str, Callable] = {}
    original_models = {}
    if models is not None:
        for fname, obj in models.items():
            model_preds[fname] = _make_model_pred(obj)
            original_models[fname] = obj
    elif evals is not None:
        fitted = _fit_evals_with_pysr(evals, function_names=function_names, sr_params=sr_params)
        for fname, obj in fitted.items():
            model_preds[fname] = _make_model_pred(obj)
            original_models[fname] = obj

    def rhs(t, y):
        var_map = {state_vars[i]: float(y[i]) for i in range(len(state_vars))}
        dydt = np.zeros_like(y, dtype=float)
        for i, expr in enumerate(rhs_exprs):
            val = _eval_sympy_symbr(expr, var_map, float(t), scalar_params, local_funcs, model_preds)
            dydt[i] = float(val)
        return dydt

    if bool(params.get('print_models', True)):
        print('=== Models used in simulation ===')
        for fname in sorted(model_preds.keys()):
            pred = model_preds[fname]
            original_model_dict = original_models.get(fname)
            expr_str = None

            # 1. First, check the original model dictionary for a pre-computed expression.
            #    This is the most direct source when using process_evals_SymbR.
            if isinstance(original_model_dict, dict):
                for key in ('expr', 'equation', 'sympy_format'):
                    if key in original_model_dict and original_model_dict[key] is not None:
                        expr_str = str(original_model_dict[key])
                        break
            
            # 2. If not found, fall back to extracting it from the PySR model object
            #    that was attached to the predictor function during its creation.
            if expr_str is None:
                pysr_model = getattr(pred, '_pymodel', None)
                if pysr_model is not None:
                    try:
                        expr_str = _extract_expr_from_pysr(pysr_model)
                    except Exception as e:
                        warnings.warn(f"Could not extract expression for {fname} from PySR model: {e}")

            # Now, print the results based on whether an expression was found
            if expr_str is not None:
                # NEW: Call the simplification function before printing
                simplified_str = _simplify_and_format_expr(expr_str)
                print(f"{fname}(x) ≈ {simplified_str}") # Use '≈' to show it's an approximation
                
                try:
                    xs = np.array(params.get('print_x_samples', [-1.0, 0.0, 1.0]), dtype=float)
                    ys = pred(xs)
                    sample_str = ', '.join([f"{x}->{y:.4g}" for x, y in zip(xs, np.asarray(ys).ravel())])
                    print(f"  samples: {sample_str}")
                except Exception as e:
                    warnings.warn(f"Could not print samples for {fname}: {e}")
            else:
                # Fallback message if no symbolic expression could be found at all
                src = getattr(pred, '_source', 'unknown')
                print(f"{fname}: No symbolic expression found. Source='{src}'")

        print('=== end models ===')


    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method=method, atol=atol, rtol=rtol)

    if check_nan and (np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y))):
        raise RuntimeError('Simulation produced NaN or Inf in solution')

    derivatives = [rhs(sol.t[i], sol.y[:, i]) for i in range(len(sol.t))]
    derivatives_array = np.array(derivatives).T

    return sol, derivatives_array


if __name__ == '__main__':
    print('This module provides simulate_SymbR(equations, params). Import and call from your script.')

