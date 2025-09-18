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
    def __init__(self, neurons=100):
        super().__init__()
        self.fc1 = nn.Linear(1, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# --- 2) Parse equation, detect f_i ---
def parse_functions(equation_str):
    # regex to find all f\d+(\w+) - Updated to use \w+ for alphanumeric args like x1, x2
    pattern = r'(f\d+)\((\w+)\)'
    funcs = re.findall(pattern, equation_str)
    # funcs: list of tuples [('f1', 'x'), ('f2', 'x_dot'), ...]
    unique_funcs = list(dict.fromkeys(funcs))  # unique preserving order
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
        if v in df.columns: # Only create tensors for columns that exist in the DataFrame
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
    tensors: dict of variable name -> torch tensor. IMPORTANT: This should be the
             dictionary containing ALL variables, including derived ones.
    
    Uses data points from tensors[var_name] for oddness check.
    """
    total_loss = 0

    for c in constraints:
        cons_str = c['constraint'].strip()
        penalty = c.get('penalty', 1.0)  # weight=1 as default
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
            if f_name in models and var_name is not None and var_name in tensors:
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
            if f_name in models and var_name is not None and var_name in tensors:
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
    MODIFIED: Now handles derived variables like x1, x2, ... defined within the equations.
    """
    if params is None:
        params = {}

    # Extract hyperparameters with defaults
    neurons = params.get('neurons', 100)
    lr = params.get('lr', 1e-3)
    epochs = params.get('epochs', 1000)
    error_threshold = params.get('error_threshold', 1e-6)
    extrapolation = params.get('extrapolation', None)
    weight_loss_param = params.get('weight_loss_param', 1.0)
    constraints = params.get('constraints', [])

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
    func_map = {}   # f_name -> var_name (ensure consistent)
    func_order = [] # list of (f_name, var_name) in appearance order
    all_symbols = set() # Keep track of all variables
    
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
        
        # Extract all symbols from the equation to find all variables
        eq_obj_temp = sympy_expression(eq)
        for sym in eq_obj_temp.free_symbols:
            all_symbols.add(str(sym))

    # collect unique scalar parameter names across all equations
    param_names_set = set()
    for eq in equations:
        for p in extract_parameters(eq):
            param_names_set.add(p)
    param_names = sorted(param_names_set)
    
    # Remove scalar params from the set of all symbols
    for p_name in param_names:
        if p_name in all_symbols:
            all_symbols.remove(p_name)
    
    # create models and scalar params
    models = {f_name: NNModel(neurons) for f_name, _ in func_order}
    scalar_params = {name: nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) for name in param_names}

    # optimizer over all parameters (scalar params + all NN params)
    full_params = list(scalar_params.values()) + [p for model in models.values() for p in model.parameters()]
    optimizer = optim.Adam(full_params, lr=lr)

    # Prepare tensors from df (only for variables present in df.columns)
    initial_variables = list(df.columns)
    tensors = prepare_tensors(df, initial_variables)

    # --- NEW LOGIC: Separate equations into definitions and loss calculations ---
    eq_objs = [sympy_expression(eq) for eq in equations]
    definition_eqs = []
    loss_eqs = []

    input_vars_from_df = set(df.columns)
    for eq, w in zip(eq_objs, weights):
        # An equation is a 'definition' if its LHS is a simple symbol
        # AND that symbol is NOT an input from the dataframe.
        if isinstance(eq.lhs, sp.Symbol) and str(eq.lhs) not in input_vars_from_df:
            definition_eqs.append(eq)
        else:
            loss_eqs.append({'lhs': eq.lhs, 'rhs': eq.rhs, 'weight': w})

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- NEW LOGIC: Compute derived variables first ---
        # Start with the tensors from the input data
        current_tensors = tensors.copy()
        
        # Sequentially evaluate the definition equations
        # NOTE: This assumes definitions don't depend on each other in a complex way
        # that would require topological sorting. A simple sequential pass is often sufficient.
        for eq in definition_eqs:
            lhs_name = str(eq.lhs)
            # Evaluate RHS using the currently available tensors
            rhs_val = evaluate_expr(eq.rhs, current_tensors, models, scalar_params)
            current_tensors[lhs_name] = rhs_val

        # --- Compute loss using the full set of tensors (initial + derived) ---
        total_data_loss = 0.0
        for eq_info in loss_eqs:
            lhs_expr = eq_info['lhs']
            rhs_expr = eq_info['rhs']
            w = eq_info['weight']
            
            lhs_val = evaluate_expr(lhs_expr, current_tensors, models, scalar_params)
            rhs_val = evaluate_expr(rhs_expr, current_tensors, models, scalar_params)
            
            # compute MSE and multiply by eq weight
            total_data_loss = total_data_loss + w * criterion(lhs_val, rhs_val)

        # constraint loss (uses the complete 'current_tensors' dict)
        constraint_loss = compute_constraint_loss(models, constraints, func_order, current_tensors)

        # L2 penalty for scalar params
        param_penalty = 0.0
        if weight_loss_param > 0 and scalar_params:
            for p in scalar_params.values():
                param_penalty += (p ** 2).mean()
            param_penalty *= weight_loss_param

        total_loss = total_data_loss + constraint_loss + param_penalty
        
        if torch.isnan(total_loss):
            print(f"Warning: NaN loss detected at epoch {epoch}. Stopping training.")
            break
            
        total_loss.backward()
        optimizer.step()

        # printing — same behavior as before but using total_data_loss
        if epoch % 100 == 0 or total_data_loss.item() < error_threshold:
            log_line = f"Epoch {epoch}, Loss: {total_data_loss.item():.2e}"
            if constraints and isinstance(constraint_loss, torch.Tensor):
                 log_line += f", Constraint: {constraint_loss.item():.2e}"
            if scalar_params:
                 log_line += ", Params: "
                 params_str = "  ".join([f"{k}: {v.item():.2e}" for k, v in scalar_params.items()])
                 log_line += params_str
            print(log_line)

        if total_data_loss.item() < error_threshold:
            print(f"Early stopping at epoch {epoch}")
            break

    # Evaluate learned functions on their variable ranges for plotting
    results = []
    for f_name, var in func_order:
        model = models[f_name]
        model.eval()
        
        # The variable for the function might be a derived one, so check in `current_tensors`
        if var in current_tensors:
            x_vals = current_tensors[var].detach().numpy().flatten()
            x_plot = np.linspace(np.min(x_vals), np.max(x_vals), 200).reshape(-1, 1).astype(np.float32)
            x_plot_tensor = torch.tensor(x_plot)
            with torch.no_grad():
                y_plot = model(x_plot_tensor).numpy().flatten()
            results.extend([x_plot.flatten(), y_plot])
        else:
            print(f"Warning: Variable '{var}' for function '{f_name}' not found for plotting.")


    return models, results, scalar_params
