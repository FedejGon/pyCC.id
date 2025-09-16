# my_library/print_cc.py



import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_equation(equation):
    # Remove spaces
    equation = equation.replace(" ", "")
    lhs, rhs = equation.split('=')
    # Extract function terms like f1(x_dot)
    function_calls = re.findall(r'f\d+\([a-zA-Z_]+\)', lhs)
    # Extract direct terms like x_ddot
    direct_terms = re.findall(r'\b[a-zA-Z_]+\b', lhs)
    direct_terms = [term for term in direct_terms if not term.startswith('f')]
    return direct_terms, function_calls, rhs


def eval_cc(
    df, 
    equation,
    models,
    numvalues=200
    ):

    # Parse equation
    direct_terms, function_calls, rhs_term = parse_equation(equation)

    # Convert DataFrame to torch tensors
    tensors = {col: torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1) for col in df.columns}

    # === Plot each learned function over the range of inputs ===
    for call in function_calls:
        match = re.match(r'f(\d+)\(([a-zA-Z_]+)\)', call)
        f_name, var = match.groups()
        model = models[f_name]
        model.eval()  # switch to evaluation mode

        # Get input range from the data
        x_vals = tensors[var].detach().numpy().flatten()
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        x_plot = np.linspace(x_min, x_max, numvalues).reshape(-1, 1).astype(np.float32)
        x_plot_tensor = torch.tensor(x_plot)

        with torch.no_grad():
            y_plot = model(x_plot_tensor).numpy()

        # Plot
        #plt.figure()
        #plt.plot(x_plot, y_plot, label=f"{f_name}({var})")
        #plt.xlabel(var)
        #plt.ylabel(f"f{f_name}({var})")
        #plt.title(f"Trained function {f_name}({var})")
        #plt.grid(True)
        #plt.legend()
        #plt.tight_layout()
        #plt.show()


        # Store results for plotting
   #     evaluations[f_name] = {
   #         "input_var": var,
   #         "x": x_plot.flatten(),
   #         "y": y_plot,
   #     }#

   # return  evaluations
    
  
    # === Evaluate and collect values for flat unpacking ===
    output_list = []

    for call in function_calls:
        match = re.match(r'f(\d+)\(([a-zA-Z_]+)\)', call)
        f_name, var = match.groups()
        model = models[f_name]
        model.eval()

        # Range of input variable
        x_vals = tensors[var].detach().numpy().flatten()
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        x_plot = np.linspace(x_min, x_max, 200).reshape(-1, 1).astype(np.float32)
        x_plot_tensor = torch.tensor(x_plot)

        with torch.no_grad():
            y_plot = model(x_plot_tensor).numpy().flatten()

        output_list.extend([x_plot.flatten(), y_plot])  # flat return

    return output_list  # Example: [x_dot, f1_vals, x, f2_vals, ...]

