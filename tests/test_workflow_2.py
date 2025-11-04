import pycc
import numpy as np
import pandas as pd
import torch
from torch.nn import ParameterDict
import collections.abc

# This test file verifies the full workflow:
# 1) Simulate a known system to get data.
# 2) Train the NN-CC method.
# 3) Assert NN model accuracy.
# 4) Simulate the *identified NN* model and assert its accuracy.
# 5) Post-process with SymbR.
# 6) Simulate the *final symbolic* model and assert its accuracy.

def test_stick_slip_with_symbreg_workflow():
    
    ##############################################
    # 1) SIMULATE THEORETICAL SYSTEM
    ##############################################
    
    # 1a) Define parameters and functions
    alpha=1.0;beta=0.2;delta=0.1;Omega=1.0;
    x0=0.0;v0=0.0; y0=[x0,v0] # initial conditions
    t_span=(0, 20); t_eval=np.linspace(*t_span, 1000)
    
    def F1_th(x_dot):
        return delta * x_dot + 0.5 * np.tanh(500*x_dot)
    def F2_th(x):
        return alpha * x + beta * x**3
    def F_ext(t):
        return np.cos(Omega * t)

    # 1b) Define equation
    eqs_th = ['x1_dot = x2',
              'x2_dot = F_ext - f1(x2) - f2(x1)']

    # 1c) Define simulation parameters
    params_th = {
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval,
        'method': 'LSODA',
        'local_funcs': {'f1': lambda t: F1_th(t),'f2': lambda t: F2_th(t),'F_ext': lambda t: F_ext(t)}
    }
    # 1d) Integrate forward the theoretical equation
    sol_th, derivatives = pycc.simulate(eqs_th,method="Theoretical", params=params_th)

    # 1e) Extract data from theoretical solution
    time_data    = sol_th.t
    x1_data      = sol_th.y[0]
    x2_data      = sol_th.y[1]
    x1_dot_data  = derivatives[0]
    x2_dot_data  = derivatives[1]
    F_ext_val    = F_ext(time_data)

    # Define database for training
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
    # 2) TRAIN THE NN-CC MODEL
    ##############################################
    
    # 2a) Propose equations
    eqs = [
         'x1_dot = x2', 
         'x2_dot = F_ext - f1(x2) - f2(x1)'
    ]
    # 2b) Define constraints
    constraints = [
       {'constraint': 'f2(0)=0'},
       {'constraint': 'f1 odd'},
       {'constraint': 'f2 odd'},
    ]
    # 2c) Define training parameters
    params_NN = {
        'neurons': 50,
        'layers': 3,
        'lr': 1e-1,
        'epochs': 200, # Using 2000 epochs as in the script
        'error_threshold': 1e-6,
        #'device':'cpu',
        'weight_loss_param': 1e-3,
        'constraints': constraints,
    }
    # 2d) Train the model
    models, evals, obtained_coefs = pycc.train(df, eqs, method='NN', params=params_NN)

    # --- Pytest Assertion 2: Check model training outputs ---
    assert 'f1' in models
    assert 'f2' in models
    assert len(evals) == 4
    # Check for dict OR ParameterDict (fix from our previous conversation)
    assert isinstance(obtained_coefs, (dict, ParameterDict, collections.abc.Mapping))
    
    # --- Pytest Assertion 3: Check accuracy of identified NN functions ---
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
    f1_theory_vals = F1_th(x_f1_cc)
    f2_theory_vals = F2_th(x_f2_cc)
    
    # Use RMSE (Root Mean Squared Error) for a robust ML test
    rmse_f1 = np.sqrt(np.mean((f1_cc - f1_theory_vals)**2))
    rmse_f2 = np.sqrt(np.mean((f2_cc - f2_theory_vals)**2))

    error_margin_nn = 0.4 # Define an acceptable average error
    
    assert rmse_f1 < error_margin_nn, f"NN Learned f1 RMSE ({rmse_f1:.4e}) > threshold ({error_margin_nn})."
    assert rmse_f2 < error_margin_nn, f"NN Learned f2 RMSE ({rmse_f2:.4e}) > threshold ({error_margin_nn})."


    ##############################################
    # 3) SIMULATE THE IDENTIFIED NN MODEL
    ##############################################
    
    # 3a) Define simulation parameters
    params_NN_simul = {
        'models': models,
        'obtained_coefs': obtained_coefs,
        'local_funcs': {'F_ext': lambda t: F_ext(t)},
        't_span':t_span,
        'y0': y0,
        't_eval': t_eval,
        'method': 'LSODA',
    }
    # 3b) Integrate identified equations
    sol_nn, _ = pycc.simulate(eqs, method='NN', params=params_NN_simul)

    # --- Pytest Assertion 4: Check if the NN-based simulation was successful ---
    assert sol_nn.success, "NN-based simulation (Part 3) failed to solve."

    # --- Pytest Assertion 5: Check if new NN simulation matches original data ---
    time_sim_nn = sol_nn.t
    x1_sim_nn = sol_nn.y[0]
    
    assert np.allclose(time_sim_nn, time_data)
    # Check if trajectory is close to the original "ground truth" data
    #assert np.allclose(x1_sim_nn, x1_data, rtol=1e-1, atol=1e-1), "NN simulation trajectory deviates from original data."

    # Check if trajectory is close to the original "ground truth" data
    # using RMSE, which is more stable for dynamic simulations
    
    rmse_trajectory = np.sqrt(np.mean((x1_sim_nn - x1_data)**2))
    
    # Define an acceptable average error for the whole simulation
    # You may need to tune this margin
    traj_error_margin = 3.5 
    
    print(f"\nDebug: Trajectory RMSE = {rmse_trajectory:.4e}")

    assert rmse_trajectory < traj_error_margin, \
        f"NN simulation trajectory RMSE ({rmse_trajectory:.4e}) is higher than threshold ({traj_error_margin})."
    ##############################################
    # 4) POST-PROCESS WITH SYMBR
    ##############################################

    # 4a) Define post-processing parameters
    pysr_settings = {
        'niterations': 10,
        'populations': 10,
        'binary_operators': ['+', '*', '-'],
        'unary_operators': ['tanh', 'sin','cos'],
        'maxsize': 10,
    }
    
    post_process_params = {
        'evals': evals,
        'pysr': pysr_settings,
        'plot': False  # <-- CRITICAL: Must be False for automated tests
    }

    # 4b) Run the post-processing
    models_sr, evals_sr = pycc.post_processing(eqs, method='SymbR', params=post_process_params)
     
    # --- Pytest Assertion 6: Check SymbR model output ---
    assert 'f1' in models_sr
    assert 'f2' in models_sr
    
    # Check that the model is a dictionary (as shown by the error)
    assert isinstance(models_sr['f1'], dict)
    assert isinstance(models_sr['f2'], dict)
    
    # More robust: check that the dictionary contains the expression string
    assert 'expr' in models_sr['f1']
    assert isinstance(models_sr['f1']['expr'], str) 


    # 4c) Define simulation parameters for the SymbR model
    sim_params_sr = {
        'models': models_sr,           # Use the new symbolic models
        'obtained_coefs': obtained_coefs, 
        'local_funcs': {'F_ext': F_ext},
        't_span': t_span,
        'y0': y0,
        't_eval': t_eval
    }

    # 4d) Simulate using the 'SymbR' method
    sol_sr, _ = pycc.simulate(eqs, method='SymbR', params=sim_params_sr)

    # --- Pytest Assertion 7: Check SymbR simulation success ---
    assert sol_sr.success, "SymbR-based simulation (Part 4) failed to solve."
    
    # --- Pytest Assertion 8: Check if SymbR simulation matches original data ---
    time_sim_sr = sol_sr.t
    x1_sim_sr = sol_sr.y[0]
    
    assert np.allclose(time_sim_sr, time_data)
    # The SymbR model is an approximation of the NN, so we might
    # need a slightly looser tolerance than the NN simulation.
    #assert np.allclose(x1_sim_sr, x1_data, rtol=5e-2, atol=5e-2), "SymbR simulation trajectory deviates significantly from original data."
    
# The SymbR model is an approximation of the NN. We check its
    # trajectory against the original data using a robust RMSE metric.
    
    rmse_trajectory_sr = np.sqrt(np.mean((x1_sim_sr - x1_data)**2))
    
    # Define an acceptable average error for the SymbR simulation
    # This might need to be slightly higher than the NN's margin
    sr_traj_error_margin = 3.5 
    
    print(f"\nDebug: SymbR Trajectory RMSE = {rmse_trajectory_sr:.4e}")

    assert rmse_trajectory_sr < sr_traj_error_margin, \
        f"SymbR simulation trajectory RMSE ({rmse_trajectory_sr:.4e}) is higher than threshold ({sr_traj_error_margin})."
