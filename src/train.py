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
    # regex to find all f\d+(\w+)
    pattern = r'(f\d+)\(([a-zA-Z_]+)\)'
    funcs = re.findall(pattern, equation_str)
    # funcs: list of tuples [('f1', 'x'), ('f2', 'x_dot'), ...]
    unique_funcs = list(dict.fromkeys(funcs))  # unique preserving order
    return unique_funcs


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

#  extract parameters a1, a2, ... to be learned in a hibrid approach
def extract_parameters(equation_str):
    return sorted(set(re.findall(r'\ba\d+\b', equation_str)))


# --- 4) Prepare PyTorch tensors from DataFrame columns ---
def prepare_tensors(df, variables):
    tensors = {}
    for v in variables:
        tensors[v] = torch.tensor(df[v].values, dtype=torch.float32).unsqueeze(1)
    return tensors


# --- 5) Evaluate sympy expression with models ---
def evaluate_expr(expr, tensors, models):
    """
    Recursively evaluate sympy expr, replacing f_i(...) with model outputs.
    tensors: dict of variable name -> tensor
    models: dict of f_i name -> nn.Module
    """
    if expr.is_Number:
        return torch.tensor(float(expr), dtype=torch.float32)

    elif expr.is_Symbol:
        var_name = str(expr)
        if var_name not in tensors:
            raise ValueError(f"Variable {var_name} not found in tensors")
        return tensors[var_name]

    elif expr.is_Function:
        # Expect expr like f1(x), f2(x_dot), etc.
        func_name = expr.func.__name__
        arg = expr.args[0]
        # Evaluate arg tensor recursively
        arg_tensor = evaluate_expr(arg, tensors, models)
        if func_name not in models:
            raise ValueError(f"Model {func_name} not found")
        return models[func_name](arg_tensor)

    elif expr.is_Add or expr.is_Mul or expr.is_Pow:
        args = [evaluate_expr(arg, tensors, models) for arg in expr.args]

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
        penalty = c.get('penalty', 1.0)
        
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
                x_pos = x_data[x_data >= 0].unsqueeze(1)
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
                x_pos = x_data[x_data >= 0].unsqueeze(1)
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


# --- 6) Training function ---
def train(df, equation_str,constraints=None, neurons=100, lr=1e-3, epochs=1000, error_threshold=1e-6,extrapolation=None):
    if constraints is None:
        constraints = []
       
    #constraints = [
    #    {'constraint': 'f1(0)=0', 'penalty': 1e-2},
    #    {'constraint': 'f2 odd', 'penalty': 1e-3},
    #]
    # Parse unique f_i functions
    func_list = parse_functions(equation_str)  # e.g. [('f1', 'x'), ...]

    # Create models for each f_i
    models = {}
    optimizers = {}

    for f_name, var in func_list:
        models[f_name] = NNModel(neurons)
        optimizers[f_name] = optim.Adam(models[f_name].parameters(), lr=lr)

    # Extract variables from df for tensors
    variables = list(df.columns)
    tensors = prepare_tensors(df, variables)

    # Parse sympy equation
    eq = sympy_expression(equation_str)
    lhs_expr = eq.lhs
    rhs_expr = eq.rhs

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Zero grads
        for opt in optimizers.values():
            opt.zero_grad()

        # Evaluate both sides
        lhs_val = evaluate_expr(lhs_expr, tensors, models)
        rhs_val = evaluate_expr(rhs_expr, tensors, models)

        loss = criterion(lhs_val, rhs_val)
        
        #constraint_loss = compute_constraint_loss(models, constraints)
        #total_loss = loss + constraint_loss
        constraint_loss = compute_constraint_loss(models, constraints, func_list, tensors)
        total_loss = loss + constraint_loss
        total_loss.backward()

        for opt in optimizers.values():
            opt.step()

        if epoch % 100 == 0 or loss.item() < error_threshold:
            print(f"Epoch {epoch}, Loss: {loss.item():.2e}")
        if loss.item() < error_threshold:
            print(f"Early stopping at epoch {epoch}")
            break

    # Evaluate models on input ranges for plotting
    results = []
    for f_name, var in func_list:
        model = models[f_name]
        model.eval()

        x_vals = tensors[var].detach().numpy().flatten()
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        x_plot = np.linspace(x_min, x_max, 200).reshape(-1, 1).astype(np.float32)
        x_plot_tensor = torch.tensor(x_plot)

        with torch.no_grad():
            y_plot = model(x_plot_tensor).numpy().flatten()

        results.extend([x_plot.flatten(), y_plot])

    return models, results

