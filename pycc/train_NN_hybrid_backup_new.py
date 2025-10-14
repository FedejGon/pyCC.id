# my_library/train_models.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import time
import re # for processing equation string
import matplotlib.pyplot as plt
import sympy as sp
# --- 1) Define NN model class ---
class NNModel(nn.Module):
    def __init__(self, neurons=100, layers=3):
        super().__init__()
        self.input_layer = nn.Linear(1, neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(layers)])
        self.output_layer = nn.Linear(neurons, 1) 
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)
        
# --- 2) Parse equation, detect f_i ---
def parse_functions(equation_str):
    # regex to find all f\d+(\w+)
    #pattern = r'(f\d+)\(([a-zA-Z_]+)\)' #without_numbers
    pattern = r'(f\d+)\((\w+)\)' # with numbers, i.e. f(x1)
    funcs = re.findall(pattern, equation_str)
    # funcs: list of tuples [('f1', 'x'), ('f2', 'x_dot'), ...]
    unique_funcs = list(dict.fromkeys(funcs)) # unique preserving order
    return unique_funcs
def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))
# --- 3) Convert string equation to sympy expression ---
def sympy_expression(equation_str):
    # Replace ^ with **
    equation_str = equation_str.replace('^', '**')
    # Parse sympy Eq object
    lhs_str, rhs_str = equation_str.split('=')
    lhs = sp.sympify(lhs_str)
    rhs = sp.sympify(rhs_str)
    eq = sp.Eq(lhs, rhs)
    return eq
# --- 4) Extract symbolic parameters like a1, a2 ---
def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))
# --- 5) Prepare PyTorch tensors from DataFrame columns ---
def prepare_tensors(df, variables):
    tensors = {}
    for v in variables:
        tensors[v] = torch.tensor(df[v].values, dtype=torch.float32).unsqueeze(1)
    return tensors
# --- 6) Evaluate sympy expression with models ---
# Create a dictionary to map sympy functions to torch functions
torch_math = {
    'exp': torch.exp,
    'cos': torch.cos,
    'sin': torch.sin,
    'log': torch.log,
    'log10': torch.log10,
    'sqrt': torch.sqrt,
    'abs': torch.abs,
    'tanh': torch.tanh,
    'tan': torch.tan,
    'cosh': torch.cosh,
    'sinh': torch.sinh,
    'acos': torch.acos,
    'asin': torch.asin,
    'atan': torch.atan,
    'atanh': torch.atanh,
    'acosh': torch.acosh,
    'asinh': torch.asinh,
    'sigmoid': torch.sigmoid,
    'erf': torch.erf,
    'erfc': torch.erfc,
    'erfinv': torch.erfinv,
    'expm1': torch.expm1,
    'log1p': torch.log1p,
    'ceil': torch.ceil,
    'floor': torch.floor,
    'trunc': torch.trunc,
    'round': torch.round,
    'sign': torch.sign,
    'sech': lambda x: 1 / torch.cosh(x),
    'csch': lambda x: 1 / torch.sinh(x),
    'coth': lambda x: torch.cosh(x) / torch.sinh(x),
}
def evaluate_expr(expr, tensors, models, scalar_params):
    if expr.is_Number:
        return torch.tensor(float(expr), dtype=torch.float32)
    elif expr.is_Symbol:
        var_name = str(expr)
        if var_name in tensors:
            return tensors[var_name]
        elif var_name in scalar_params:
            return scalar_params[var_name]
        else:
            raise ValueError(f"Unknown symbol {var_name}")
    elif expr.is_Function:
        func_name = expr.func.__name__
        arg = expr.args[0]
        arg_tensor = evaluate_expr(arg, tensors, models, scalar_params)
        
        # Check if it's a learned function (f1, f2, etc.)
        if func_name in models:
            return models[func_name](arg_tensor)
        # Check if it's a standard mathematical function
        elif func_name in torch_math:
            return torch_math[func_name](arg_tensor)
        else:
            raise ValueError(f"Model or known function '{func_name}' not found")
    elif expr.is_Add or expr.is_Mul or expr.is_Pow:
        args = [evaluate_expr(arg, tensors, models, scalar_params) for arg in expr.args]
        if expr.is_Add:
            return sum(args)
        elif expr.is_Mul:
            result = args[0]
            for a in args[1:]:
                result = result * a
            return result
        elif expr.is_Pow:
            base, exp = args
            return base ** exp
    else:
        raise NotImplementedError(f"Expr type {expr} not implemented")
# --- 7) Optional constraints ---
def compute_constraint_loss(models, constraints, func_list, tensors):
    """
    models: dict of function name -> nn.Module
    constraints: list of dicts like {'constraint': str, 'penalty': float}
    func_list: list of (f_name, var_name) tuples, e.g. [('f1', 'x_dot'), ('f2', 'x')]
    tensors: dict of variable name -> torch tensor
    
    Uses data points from tensors[var_name] for oddness check.
    """
    total_loss = 0
    for c in constraints:
        cons_str = c['constraint'].strip()
        penalty = c.get('penalty', 1.0) # weight=1 as default
        mode = c.get('eval','array') # eval can be 'data' or 'array'
        Nval_array = c.get('Nval_array',100)
        
        # --- Match value-at-point: e.g. f1(3) = 2.5 ---
        m_val_at = re.match(r'(f\d+)\(([-+]?\d*\.?\d+)\)\s*=\s*([-+]?\d*\.?\d+)', cons_str)
        if m_val_at:
            f_name = m_val_at.group(1)
            x_val = float(m_val_at.group(2))
            y_target = float(m_val_at.group(3))
            if f_name in models:
                model = models[f_name]
                x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
                y_pred = model(x_tensor)
                loss = ((y_pred - y_target) ** 2).mean()
                total_loss += penalty * loss
            continue
        
        # --- Match odd function constraint: e.g. "f2 odd" ---
        m_odd = re.match(r'(f\d+)\s+odd', cons_str)
        if m_odd:
            f_name = m_odd.group(1)
            # Find var_name from func_list
            var_name = None
            for fn, vn in func_list:
                if fn == f_name:
                    var_name = vn
                    break
            if f_name in models and var_name is not None:
                model = models[f_name]
                x_data = tensors[var_name].flatten()
                if mode=='data':
                    #print(f"Constraint data for {f_name}")
                    # simple implementation where we evaluate the data points
                    x_pos = x_data[x_data >= 0].unsqueeze(1)
                elif mode=='array':
                    # create a linspace to evaluate an equispaced interval
                    #print(f"Constraint array for {f_name}")
                    max_abs_val = x_data.abs().max()
                    x_pos = torch.linspace(0.0, max_abs_val, steps=Nval_array, device=x_data.device).unsqueeze(1)
                else:
                    print(f"Warning: eval mode of constraint '{m_odd}' not recognized.")
                x_neg = -x_pos
                y_pos = model(x_pos)
                y_neg = model(x_neg)
                loss = ((y_pos + y_neg) ** 2).mean()
                total_loss += penalty * loss
            continue
        # --- Match even function constraint: e.g. "f2 even" ---
        m_even = re.match(r'(f\d+)\s+even', cons_str)
        if m_even:
            f_name = m_even.group(1)
            # Find var_name from func_list
            var_name = None
            for fn, vn in func_list:
                if fn == f_name:
                    var_name = vn
                    break
            if f_name in models and var_name is not None:
                model = models[f_name]
                x_data = tensors[var_name].flatten()
                if mode=='data':
                    # simple implementation where we evaluate the data points
                    x_pos = x_data[x_data >= 0].unsqueeze(1)
                elif mode=='array':
                    # create a linspace to evaluate an equispaced interval
                    max_abs_val = x_data.abs().max()
                    x_pos = torch.linspace(0.0, max_abs_val, steps=Nval_array, device=x_data.device).unsqueeze(1)
                else:
                    print(f"Warning: eval mode of constraint '{m_even}' not recognized.")
                x_neg = -x_pos
                y_pos = model(x_pos)
                y_neg = model(x_neg)
                loss = ((y_pos - y_neg) ** 2).mean()
                total_loss += penalty * loss
            continue
        
        
        # --- You can add more constraint types here ---
        
        else:
            # Unknown constraint string
            print(f"Warning: constraint '{cons_str}' not recognized.")
    
    return total_loss
# --- 8) Training function ---
def train_NN_hybrid(df, equation_str, params=None):
    """
    Backwards-compatible trainer that accepts one equation (str) or multiple (list/tuple).
    Added: per-equation weights via params['eq_weights'] (default 1.0 for each equation).
    """
    if params is None:
        params = {}
    # Extract hyperparameters with defaults
    neurons = params.get('neurons', 100)
    layers = params.get('layers', 3)
    lr = params.get('lr', 1e-3)
    epochs = params.get('epochs', 1000)
    error_threshold = params.get('error_threshold', 1e-6)
    extrapolation = params.get('extrapolation', None)
    weight_loss_param = params.get('weight_loss_param', 1e-3)
    constraints = params.get('constraints', [])
    n_eval = int(params.get('n_eval', 200))
    # Accept single equation or list of equations (backward compatible)
    if isinstance(equation_str, (list, tuple)):
        equations = list(equation_str)
    else:
        equations = [equation_str]
    # Handle eq_weights from params
    eq_weights = params.get('eq_weights', None)
    if eq_weights is None:
        weights = [1.0] * len(equations)
    elif np.isscalar(eq_weights):
        weights = [float(eq_weights)] * len(equations)
    else:
        # assume list/tuple-like
        if len(eq_weights) != len(equations):
            raise ValueError(f"eq_weights length {len(eq_weights)} does not match number of equations {len(equations)}")
        weights = [float(w) for w in eq_weights]
    # Parse functions and parameters from all equations, preserving order and ensuring consistency
    func_map = {} # f_name -> var_name (ensure consistent)
    func_order = [] # list of (f_name, var_name) in appearance order
    for eq in equations:
        flist = parse_functions(eq)
        for f_name, var_name in flist:
            if f_name in func_map:
                if func_map[f_name] != var_name:
                    raise ValueError(f"Function {f_name} used with different arguments: "
                                     f"{func_map[f_name]} vs {var_name}")
            else:
                func_map[f_name] = var_name
                func_order.append((f_name, var_name))
    # collect unique scalar parameter names across all equations
    param_names_set = set()
    for eq in equations:
        for p in extract_parameters(eq):
            param_names_set.add(p)
    param_names = sorted(param_names_set)
    # create models and scalar params
    models = {f_name: NNModel(neurons,layers) for f_name, _ in func_order}
    scalar_params = {name: nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) for name in param_names}
    # optimizer over all parameters (scalar params + all NN params)
    full_params = list(scalar_params.values()) + [p for model in models.values() for p in model.parameters()]
    optimizer = optim.Adam(full_params, lr=lr)
    # Prepare tensors from df (same as before)
    variables = list(df.columns)
    tensors = prepare_tensors(df, variables)
    # Parse every equation into sympy Eq objects
    eq_objs = [sympy_expression(eq) for eq in equations]
    lhs_exprs = [eqo.lhs for eqo in eq_objs]
    rhs_exprs = [eqo.rhs for eqo in eq_objs]
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Compute total loss as weighted sum over all equations' MSE(lhs, rhs)
        total_data_loss = 0.0
        for lhs_expr, rhs_expr, w in zip(lhs_exprs, rhs_exprs, weights):
            lhs_val = evaluate_expr(lhs_expr, tensors, models, scalar_params)
            rhs_val = evaluate_expr(rhs_expr, tensors, models, scalar_params)
            # compute MSE and multiply by eq weight
            total_data_loss = total_data_loss + w * criterion(lhs_val, rhs_val)
        # constraint loss (same helper, uses func_order and tensors)
        constraint_loss = compute_constraint_loss(models, constraints, func_order, tensors)
        # L2 penalty for scalar params
        param_penalty = 0.0
        if weight_loss_param > 0:
            for p in scalar_params.values():
                param_penalty += (p ** 2).mean()
            param_penalty *= weight_loss_param
        total_loss = total_data_loss + constraint_loss + param_penalty
        total_loss.backward()
        optimizer.step()
        # printing — same behavior as before but using total_data_loss
        if epoch % 100 == 0 or total_data_loss.item() < error_threshold:
            if scalar_params and constraints:
                print(f"Epoch {epoch}, Loss: {total_data_loss.item():.2e}, Constraint: {constraint_loss:.2e}, Params: ", end="")
                for k in scalar_params:
                    print(f"{k}: {scalar_params[k].item():.2e}", end="  ")
                print()
            elif scalar_params:
                print(f"Epoch {epoch}, Loss: {total_data_loss.item():.2e}, Params: ", end="")
                for k in scalar_params:
                    print(f"{k}: {scalar_params[k].item():.2e}", end="  ")
                print()
            elif constraints:
                print(f"Epoch {epoch}, Loss: {total_data_loss.item():.2e}, Constraint: {constraint_loss:.2e}")
            else:
                print(f"Epoch {epoch}, Loss: {total_data_loss.item():.2e}")
        if total_data_loss.item() < error_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
    # Evaluate learned functions on their variable ranges for plotting (same as before)
    results = []
    for f_name, var in func_order:
        model = models[f_name]
        model.eval()
        x_vals = tensors[var].detach().numpy().flatten()
        x_plot = np.linspace(np.min(x_vals), np.max(x_vals), n_eval).reshape(-1, 1).astype(np.float32)
        x_plot_tensor = torch.tensor(x_plot)
        with torch.no_grad():
            y_plot = model(x_plot_tensor).numpy().flatten()
        results.extend([x_plot.flatten(), y_plot])
    return models, results, scalar_params
