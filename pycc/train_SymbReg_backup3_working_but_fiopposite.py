import numpy as np
import re
import sympy as sp
from collections import OrderedDict
from pysr import PySRRegressor

###############################################
# --- Parsing utilities ---
###############################################

def parse_functions(equation_str):
    pattern = r'(f\d+)\(([a-zA-Z_]+)\)'
    all_funcs = re.findall(pattern, equation_str)
    unique_funcs = list(OrderedDict.fromkeys(all_funcs))
    return unique_funcs

def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))

###############################################
# --- Symbolic Regression Trainer ---
###############################################

def train_SymbReg(df, equation, params=None):
    if params is None:
        params = {}

    # PySR params
    niterations = int(params.get('niterations', 100))
    populations = int(params.get('populations', 10))
    eq_weights = params.get('eq_weights', None)

    equations = list(equation) if isinstance(equation, (list, tuple)) else [equation]
    N_data = len(df)

    # --- Parse functions and scalar parameters ---
    func_map = OrderedDict()
    param_names = set()
    for eq in equations:
        for f_name, var_name in parse_functions(eq):
            if f_name in func_map and func_map[f_name] != var_name:
                raise ValueError(f"Function {f_name} used with different arguments.")
            func_map[f_name] = var_name
        param_names.update(extract_parameters(eq))

    func_order = list(func_map.items())
    param_names = sorted(list(param_names))

    # --- Symbolic preparation ---
    sym_exprs = []
    for eq_str in equations:
        lhs_str, rhs_str = eq_str.split('=', 1)
        expr = sp.sympify(lhs_str) - sp.sympify(rhs_str)
        sym_exprs.append(expr)

    f_syms = {f_name: sp.Symbol(f_name) for f_name, _ in func_order}
    a_syms = {a_name: sp.Symbol(a_name) for a_name in param_names}
    func_call_to_sym_map = {sp.Function(f_name)(sp.Symbol(var_name)): f_syms[f_name] for f_name, var_name in func_order}

    lambdas = {}
    for i, expr in enumerate(sym_exprs):
        expr_simple = expr.subs(func_call_to_sym_map)
        all_needed_syms = sorted(expr_simple.free_symbols, key=lambda s: s.name)
        lambdas[f'R_{i}'] = sp.lambdify(all_needed_syms, expr_simple, 'numpy')
        lambdas[f'vars_{i}'] = all_needed_syms

    # --- Train each f with symbolic regression ---
    models = {}
    evals = []

    for f_name, var_name in func_order:
        print(f"\nTraining {f_name}({var_name}) with Symbolic Regression...")

        x = df[var_name].values.astype(float)

        # Target for f_name: rearrange equation residuals
        # Build residual with current guess (f=0)
        residuals = []
        for i, expr in enumerate(sym_exprs):
            lambda_args = {}
            for sym in lambdas[f'vars_{i}']:
                s_name = sym.name
                if s_name in df.columns:
                    lambda_args[s_name] = df[s_name].values
                elif s_name in f_syms:
                    # set f_syms except the one being trained to zero for now
                    lambda_args[s_name] = np.zeros(N_data)
                elif s_name in a_syms:
                    lambda_args[s_name] = np.zeros(N_data)
                else:
                    lambda_args[s_name] = np.zeros(N_data)

            res_i = lambdas[f'R_{i}'](**{s.name: lambda_args[s.name] for s in lambdas[f'vars_{i}']})
            residuals.append(res_i)

        # Combine residuals as target for regression
        if len(residuals) > 1:
            target_y = np.mean(np.vstack(residuals), axis=0)
        else:
            target_y = residuals[0]

        # Train with PySR
        model = PySRRegressor(
            niterations=niterations,
            populations=populations,
            binary_operators=["+", "-", "*"],
            unary_operators=["sin", "cos", "exp", "log"],
            procs=0,
            progress=True,
        )
        model.fit(x.reshape(-1, 1), target_y)

        models[f_name] = model

        # Evaluation for plotting
        x_plot = np.linspace(x.min(), x.max(), 200)
        y_plot = model.predict(x_plot.reshape(-1, 1))
        evals.extend([x_plot, y_plot])

    # No scalar fitting yet, scalars left as empty dict
    final_scalars = {}

    return models, evals, final_scalars

###############################################
# Example usage (if run directly)
###############################################
if __name__ == "__main__":
    import pandas as pd

    # Example dataset
    N = 200
    t = np.linspace(0, 10, N)
    x = np.sin(t)
    x_dot = np.cos(t)
    F_ext = np.sin(2*t)
    x_ddot = -np.sin(t)

    df = pd.DataFrame({"t": t, "x": x, "x_dot": x_dot, "x_ddot": x_ddot, "F_ext": F_ext})

    equation = "x_ddot = F_ext - f1(x_dot) - f2(x)"

    models, evals, scalars = train_symbolic(df, equation, params={"niterations": 50})

    print("\nTrained models:")
    for f_name, model in models.items():
        print(f"{f_name}: {model}")

