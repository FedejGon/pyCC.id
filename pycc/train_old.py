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

# Build a single-layer or multi-layer MLP with functions
#def build_model(neurons=100):
#    layers = nn.Sequential(
#        nn.Linear(1, neurons),
#        nn.ReLU(),
#        nn.Linear(neurons, neurons),
#        nn.ReLU(),
#        nn.Linear(neurons, 1)
#    )
#    return layers

def build_model(neurons=100):
    class NN(nn.Module):
        def __init__(self):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(1, neurons)
            self.fc2 = nn.Linear(neurons, neurons)
            self.fc3 = nn.Linear(neurons, neurons)
            self.fc4 = nn.Linear(neurons, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

    return NN()


def train(
    df, 
    equation,
    neurons=100,
    learning_rate=1e-3,
    epochs_max=5000,
    error_threshold=1e-8,
    lambda_penalty=1.0,
    apply_restriction=True,
    extrapolation=None,
):

    # Parse equation
    direct_terms, function_calls, rhs_term = parse_equation(equation)
    print("Direct terms:", direct_terms)
    print("Function calls:", function_calls)
    print("RHS term:", rhs_term)
    # Convert DataFrame to torch tensors
    tensors = {col: torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1) for col in df.columns}

    # Define models for each function fi(...)
    models = {}
    optimizers = {}

    for call in function_calls:
        match = re.match(r'f(\d+)\(([a-zA-Z_]+)\)', call)
        f_name, var = match.groups()
        model = build_model(neurons)
        models[f_name] = model
        optimizers[f_name] = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    time_start = time.time()

    for epoch in range(epochs_max):
        # Build LHS = direct_term + sum(fi(input_var))
        lhs = torch.zeros_like(tensors[direct_terms[0]])
        for term in direct_terms:
            lhs += tensors[term]

        for call in function_calls:
            match = re.match(r'f(\d+)\(([a-zA-Z_]+)\)', call)
            f_name, var = match.groups()
            model = models[f_name]
            lhs += model(tensors[var])

        rhs = tensors[rhs_term]

        loss = criterion(lhs, rhs)

        # Constraint: f(0) = 0
        if apply_restriction:
            zero_input = torch.tensor([[0.0]], dtype=torch.float32)
            constraint_loss = sum((model(zero_input) ** 2).mean() for model in models.values())
            total_loss = loss + lambda_penalty * constraint_loss
        else:
            total_loss = loss

        # Backpropagation
        for opt in optimizers.values():
            opt.zero_grad()
        total_loss.backward()
        for opt in optimizers.values():
            opt.step()

        # Logging
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print("LHS sample:", lhs[:5].flatten())
            print("RHS sample:", rhs[:5].flatten())
            print(f"Epoch [{epoch+1}], Loss: {total_loss.item():.4e}, Constraint: {constraint_loss.item():.4e}")

        if total_loss.item() < error_threshold:
            print(f"Early stopping at epoch {epoch+1}, Loss: {total_loss.item():.4e}")
            break

    time_end = time.time()
    print(f"Training time: {time_end - time_start:.2f} seconds")
    
    
    
    # === Plot each learned function over the range of inputs ===
    for call in function_calls:
        match = re.match(r'f(\d+)\(([a-zA-Z_]+)\)', call)
        f_name, var = match.groups()
        model = models[f_name]
        model.eval()  # switch to evaluation mode

        # Get input range from the data
        x_vals = tensors[var].detach().numpy().flatten()
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        x_plot = np.linspace(x_min, x_max, 200).reshape(-1, 1).astype(np.float32)
        x_plot_tensor = torch.tensor(x_plot)

        with torch.no_grad():
            y_plot = model(x_plot_tensor).numpy()

        # Plot
        plt.figure()
        plt.plot(x_plot, y_plot, label=f"{f_name}({var})")
        plt.xlabel(var)
        plt.ylabel(f"f{f_name}({var})")
        plt.title(f"Trained function {f_name}({var})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    

    return models  # returns a dict like {'f1': model1, 'f2': model2}

