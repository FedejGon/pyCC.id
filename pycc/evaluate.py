# my_library/train_models.py



import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train(
    t_simul,
    x_data,
    x_dot_data,
    x_ddot_data,
    F_ext_val,
    neurons=100,
    learning_rate=1e-3,
    epochs_max=5000,
    error_threshold=1e-8,
    lambda_penalty=1.0,
    apply_restriction=True,
    f1_symmetry=None,
    f2_symmetry=None
):
    # Prepare data
    t_tensor = torch.tensor(t_simul, dtype=torch.float32).unsqueeze(1)
    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
    x_dot_tensor = torch.tensor(x_dot_data, dtype=torch.float32).unsqueeze(1)
    x_ddot_tensor = torch.tensor(x_ddot_data, dtype=torch.float32).unsqueeze(1)
    F_ext_tensor = torch.tensor(F_ext_val, dtype=torch.float32).unsqueeze(1)

    # Define NN1
    class NN1(nn.Module):
        def __init__(self):
            super(NN1, self).__init__()
            self.fc1 = nn.Linear(1, neurons)
            self.fc2 = nn.Linear(neurons, neurons)
            self.fc3 = nn.Linear(neurons, neurons)
            self.fc4 = nn.Linear(neurons, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

    # Define NN2
    class NN2(nn.Module):
        def __init__(self):
            super(NN2, self).__init__()
            self.fc1 = nn.Linear(1, neurons)
            self.fc2 = nn.Linear(neurons, neurons)
            self.fc3 = nn.Linear(neurons, neurons)
            self.fc4 = nn.Linear(neurons, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

    # Initialize models and optimizers
    model1 = NN1()
    model2 = NN2()
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

    # Training loop
    time_start = time.time()
    for epoch in range(epochs_max):
        model1.train()
        model2.train()

        predictions = x_ddot_tensor + model1(x_dot_tensor) + model2(x_tensor)
        loss = criterion(predictions, F_ext_tensor)

        if apply_restriction:
            zero_input = torch.tensor([[0.0]], dtype=torch.float32)
            constraint_loss = lambda_penalty * (
                (model1(zero_input) ** 2).mean() + (model2(zero_input) ** 2).mean()
            )
            total_loss = loss + constraint_loss
        else:
            constraint_loss = torch.tensor(0.0)
            total_loss = loss

        # Backprop
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraint: {constraint_loss.item():.4e}")
        if total_loss.item() < error_threshold:
            print(f"Training stopped at epoch {epoch}, Total Loss: {total_loss.item()}")
            break

    time_end = time.time()
    print(f"Neurons: {neurons}")
    print(f"Training time: {time_end - time_start:.2f} seconds")

    return model1, model2

