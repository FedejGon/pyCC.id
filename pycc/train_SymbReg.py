# my_library/train_models_SR.py

import sympy as sp
import numpy as np
import re
from pysr import PySRRegressor

# --- 1) Evaluate sympy expr using current PySR models ---
def evaluate_expr_SR(expr, df, models):
    """Evaluate a sympy expr with known f_i models."""
    if expr.is_Number:
        return np.full(len(df), float(expr))
    elif expr.is_Symbol:
        var_name = str(expr)
        if var_name in df.columns:
            return df[var_name].values
        else:
            raise ValueError(f"Unknown symbol {var_name}")
    elif expr.is_Function:
        func_name = expr.func.__name__
        arg = expr.args[0]
        arg_vals = evaluate_expr_SR(arg, df, models)
        if func_name not in models:
            raise ValueError(f"Model {func_name} not found")
        return models[func_name](arg_vals.reshape(-1, 1))
    elif expr.is_Add or expr.is_Mul or expr.is_Pow:
        args = [evaluate_expr_SR(arg, df, models) for arg in expr.args]
        if expr.is_Add:
            return sum(args)
        elif expr.is_Mul:
            result = args[0]
            for a in args[1:]:
                result = result * args[1]
            return result
        elif expr.is_Pow:
            base, exp = args
            return base ** exp
    else:
        raise NotImplementedError(f"Expr type {expr} not implemented")


# --- 2) Main SR training function ---
def train_SymbReg(df, equation_str, params=None):
    """
    Train f_i functions iteratively using PySR symbolic regression.
    Returns models dict, evaluation points, empty scalar_coefs dict (for compatibility).
    """
    if params is None:
        params = {}
    pysr_params = params.get('pysr', {})
    N_fit_points = params.get('N_fit_points', 200)
    num_iterations = params.get('num_iterations', 5)

    # Parse f_i functions
    func_list = re.findall(r'(f\d+)\(([a-zA-Z_][a-zA-Z0-9_]*)\)', equation_str)
    func_order = list(dict.fromkeys(func_list))  # preserve order

    # Initialize models dict
    models = {f_name: lambda x: np.zeros_like(x) for f_name, _ in func_order}

    # Sympify equation
    equation_sym = sp.sympify(equation_str.replace('^', '**'))
    if isinstance(equation_sym, sp.Equality):
        equation_sym = equation_sym.lhs - equation_sym.rhs

    # Iterative SR loop
    for it in range(num_iterations):
        print(f"\n--- Iteration {it + 1}/{num_iterations} ---")
        for f_name, var in func_order:
            print(f"-> Fitting {f_name}({var}) ...")

            # Compute target residual for current f_i
            residual = equation_sym
            for other_name, other_var in func_order:
                if other_name == f_name:
                    continue
                residual = residual.subs(sp.Function(other_name)(sp.Symbol(other_var)),
                                         evaluate_expr_SR(sp.Function(other_name)(sp.Symbol(other_var)), df, models))

            # Lambdify remaining symbols (should be only var)
            target_vals = evaluate_expr_SR(residual, df, models)

            # Fit PySR
            X = df[var].values.reshape(-1, 1)
            model_i = PySRRegressor(**pysr_params)
            model_i.fit(X, target_vals, variable_names=[var])
            expr_i = model_i.sympy()
            fun_i = sp.lambdify(sp.Symbol(var), expr_i, "numpy")

            models[f_name] = fun_i
            print(f"  {f_name}({var}) = {expr_i}")

    # Build evaluation points for plotting
    evals = []
    for f_name, var in func_order:
        x_vals = df[var].values
        x_plot = np.linspace(np.min(x_vals), np.max(x_vals), N_fit_points)
        y_plot = models[f_name](x_plot.reshape(-1, 1))
        evals.extend([x_plot, y_plot])

    scalar_coefs = {}  # for compatibility with NN version
    return models, evals, scalar_coefs

