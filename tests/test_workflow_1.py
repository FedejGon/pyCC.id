# Import the package into your Python environment
import torch
from torch.nn import ParameterDict
import pycc
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Not needed for an automated test

# This test file verifies the full workflow:
# 1) Simulate a known system to get data.
# 2) Train the NN-CC method to identify the model.
# 3) Assert that the identified functions are accurate.
# 4) Simulate the *identified* model.
# 5) Assert that the new simulation matches the original data.

def test_stick_slip_full_workflow():
    
    ##############################################
    # 1) how to simulate a stick-slip second order system using pycc.simulate()
    # 1a) define parameters and functions
    alpha=1.0;beta=0.2;delta=0.1;Omega=1.0;
    x0=0.0;v0=0.0; y0=[x0,v0] # initial conditions
    t_span=(0, 20); t_eval=np.linspace(*t_span, 1000)
    
    def F1_th(x_dot):
        return delta * x_dot + 0.5 * np.tanh(500*x_dot)
    def F2_th(x):
        return alpha * x + beta * x**3
    def F_ext(t):
        return np.cos(Omega * t)
    
    # 1b) define equation
    eqs_th = ['x1_dot = x2',
              'x2_dot = F_ext - f1(x2) - f2(x1)']
    
    # 1c) define simulation parameters
    params_th = {
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
        'method': 'LSODA',
        'local_funcs': {'f1': lambda t: F1_th(t),'f2': lambda t: F2_th(t),'F_ext': lambda t: F_ext(t)}
    }
    # 1d) integrate forward the theoretical equation
    sol,derivatives = pycc.simulate(eqs_th,method="Theoretical", params=params_th)
    
    # 1e) extract data from theoretical solution
    time_data    = sol.t
    x1_data      = sol.y[0]
    x2_data      = sol.y[1]
    x1_dot_data  = derivatives[0]
    x2_dot_data  = derivatives[1]
    F_ext_val    = F_ext(time_data)
    
    # define database for training
    df = pd.DataFrame({
        'x1':x1_data,
        'x2':x2_data,
        'x1_dot':x1_dot_data,
        'x2_dot':x2_dot_data,
        'F_ext': F_ext_val
    })

    # --- Pytest Assertion 1: Check that data was generated ---
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == len(t_eval)

    ##############################################
    # 2) how to train the NN-CC method to identify the model [pycc.train()]
    # 2a) propose equations to use for identification (fi functions and ai parameters).
    eqs = [
         'x1_dot = x2',
         'x2_dot = F_ext - f1(x2) - f2(x1)'
    ]
    # 2b) define constraints (optional)
    constraints = [ # adding prior known information
       {'constraint': 'f2(0)=0'},
       {'constraint': 'f1 odd'},
       {'constraint': 'f2 odd'},
    ]
    # 2c) define training parameters (optional)
    params_NN = {
        'neurons': 100,
        'layers':3,
        'lr': 1e-4,
        'epochs': 2000,
        'error_threshold': 1e-6,
        'extrapolation': None,
        'device':'cpu',
        'weight_loss_param': 1e-3,
        'constraints': constraints,
    }
    # 2d) train/fit/identify the model
    models, evals, obtained_coefs = pycc.train(df, eqs,method='NN', params=params_NN)

    # --- Pytest Assertion 2: Check model training outputs ---
    assert 'f1' in models  # Check that the models were created
    assert 'f2' in models
    assert len(evals) == 4    # Check for x, y for f1 and x, y for f2
    #assert isinstance(obtained_coefs, dict) # Check that coefficients dict is returned
    assert isinstance(obtained_coefs, (dict, ParameterDict))
    # --- Pytest Assertion 3: Check accuracy of identified functions ---
    # This is the most important test.
    # We check if the learned functions (f1_cc, f2_cc) are
    # numerically close to the theoretical functions (F1_th, F2_th).
    
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
    
    # Calculate the "ground truth" values at the same evaluation points
    f1_theory_vals = F1_th(x_f1_cc)
    f2_theory_vals = F2_th(x_f2_cc)
    # NEW: Calculate Root Mean Squared Error (RMSE)
    rmse_f1 = np.sqrt(np.mean((f1_cc - f1_theory_vals)**2))
    rmse_f2 = np.sqrt(np.mean((f2_cc - f2_theory_vals)**2))

    # Define an acceptable average error margin
    # You can tune this threshold. Lower is better.
    error_margin = 0.1 

    print(f"\nDebug: f1 RMSE = {rmse_f1:.4e}")
    print(f"Debug: f2 RMSE = {rmse_f2:.4e}")

    assert rmse_f1 < error_margin, f"Learned f1 function RMSE ({rmse_f1:.4e}) is higher than threshold ({error_margin})."
    assert rmse_f2 < error_margin, f"Learned f2 function RMSE ({rmse_f2:.4e}) is higher than threshold ({error_margin})."
    
    # Use numpy.allclose to check if arrays are close within a tolerance
    # (You can adjust rtol (relative) and atol (absolute) as needed)
    #assert np.allclose(f1_cc, f1_theory_vals, rtol=0.1, atol=0.1), "Learned f1 function is not close to theoretical."
    #assert np.allclose(f2_cc, f2_theory_vals, rtol=0.1, atol=0.1), "Learned f2 function is not close to theoretical."

    ##############################################
    # 3) how to simulate the identified model [pycc.simulate()]
    
    # 3a) define simulation parameters
    params_NN_simul = {
        'models': models,
        'obtained_coefs': obtained_coefs,
        'local_funcs': {'F_ext': lambda t: F_ext(t)},
        't_span':t_span,
        'y0': y0,
        't_eval': t_eval,
        'method': 'LSODA',  # solve_ivp
        'atol': 1e-8,
        'rtol': 1e-6,
        'check_nan': True
    }
    # 3b) integrate identified equations
    sol_nn, _ = pycc.simulate(eqs, method='NN', params=params_NN_simul)
    
    # --- Pytest Assertion 4: Check if the NN-based simulation was successful ---
    assert sol_nn.success, "NN-based simulation failed to solve."

    # --- Pytest Assertion 5: Check if new simulation matches original data ---
    time_sim = sol_nn.t
    x1_sim = sol_nn.y[0]
    
    # Check if the new simulation trajectory is close to the original "ground truth" data
    assert np.allclose(time_sim, time_data)
    assert np.allclose(x1_sim, x1_data, rtol=1e-2, atol=1e-2), "NN simulation trajectory deviates from original data."
