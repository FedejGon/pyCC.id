import pycc
import matplotlib.pyplot as plt # for figures
import numpy as np # used here to define math functions
import pandas as pd # used here to generate databases


# Integration of the theoretical EDO
# To this aim, we may use standard python integrators, but in this example we
# will show how to use the pycc library to integrate the theoretical EDO.
#
# define parameters
alpha=1.0; beta=0.2; delta=0.1; mu=0.5; omega=1.0; A=1.0
t_span  = (0, 20)
t_eval  = np.linspace(*t_span, 1000)
x0     = 0.0; v0     = 0.0
y0      = [x0, v0]  # initial conditions
# Theoretical functions
def F1(x_dot):
    return delta * x_dot + mu * np.tanh(500*x_dot)
def F2(x):
    return alpha * x + beta * x**3
def F_ext(t):
    return A * np.cos(omega * t)


# In the following:
# 1) define the equations we want to integrate,  (functions must be fi and params ai)
# 2) define parameters for the integrator,
# 3) we call to pycc to make the theoretical simulation
equations = [
    'x1_dot = x2',
    'x2_dot = F_ext - f1(x2) - f2(x1)'
]
params_th = {
    # we internally use ivp_solve python function with params defined in the following 4 lines
    't_span': t_span,
    'y0': y0,
    't_eval': t_eval,
    'method': 'LSODA',
    # we need to tell to pycc how to find external/local functions
    'local_funcs': {'f1': lambda x: F1(x),'f2': lambda x: F2(x),'F_ext': lambda t: F_ext(t)},
    'scalar_params': {'a1':2.0}
}
# now use pycc to integrate the EDOs
sol,derivatives = pycc.simulate(equations,method="Theoretical", params=params_th)
# Extract results
time_data   = sol.t # sol is the OdeSolution returned from ivp_solve
x1_data     = sol.y[0] # x1=x(t)
x2_data     = sol.y[1] # x2=x_dot(t)
F_ext_val   = F_ext(time_data)
x1_dot_data = derivatives[0] #x1_dot=x_dot(t)
x2_dot_data = derivatives[1] #x2_dot=x_ddot(t)
#F1_th = F1(x2_data)
#F2_th = F2(x1_data)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].plot(time_data, x1_data, label='$x_1$ (Position)', color='blue', linewidth=2)
axes[0].plot(time_data, x1_dot_data, label='$\\dot{x}_1=x_2$ (Velocity)', color='red', linestyle='--', linewidth=2)
axes[0].set_title('Position and Velocity vs. Time', fontsize=16)
axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('Value', fontsize=12)
axes[0].legend()
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[1].plot(time_data, x2_dot_data, label='$\\dot{x}_2$ (State Derivative)', color='orange', linestyle='--', linewidth=2)
axes[1].plot(time_data, F_ext_val, label='$F_{ext}$ (External Force)', color='purple', linestyle=':', linewidth=2)
axes[1].set_title('$x_2$, $\\dot{x}_2$ and External Force vs. Time', fontsize=16)
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Value', fontsize=12)
axes[1].legend()
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()




# define the proposed equatoin and generate database for identification
eqs = [
    'x1_dot = x2*(a1-1.0)',
    'x2_dot = F_ext - f1(x2) - f2(x1)'
]

eqs = [
    'x1_dot = x2*(a1-1.0)',
    'x2_dot = F_ext - a2*x2-a3*tanh(500*a4*x2) - a5*x1-a6*x1**3'
]

#alpha=1.0; beta=0.2; delta=0.1; mu=0.5; omega=1.0; A=1.0
#def F1(x_dot):
#    return delta * x_dot + mu * np.tanh(500*x_dot)
#def F2(x):
#    return alpha * x + beta * x**3




df = pd.DataFrame({
    #'t': time_data, # this can be added if eqs include t as an explicit variable
    'x1':x1_data,
    'x2':x2_data,
    'x1_dot':x1_dot_data,
    'x2_dot':x2_dot_data,
    'F_ext': F_ext_val
})



# METHOD 2:  using neural networks (NN-CC)
constraints = [ # adding prior known information
   {'constraint': 'f2(0)=0'},
   {'constraint': 'f1 odd'},
   {'constraint': 'f2 odd'},
]
parameters_NN = {
    'neurons': 100,
    'layers':3,
    'lr': 1e-3,
    'scalar_lr': 1e-3, 
    'epochs': 10000,
    'error_threshold': 1e-6,
    'extrapolation': None,
    'device':'cpu', #'cpu', 'gpu'
    'weight_loss_param': 1e-3,
    'constraints': constraints,
}

models, evals, obtained_coefs = pycc.train(df, eqs,method='NN', params=parameters_NN)


# plot obtained functions (characteristic curves CCs)
x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(x_f1_cc, f1_cc, label='$f_1$ learned NN-CC')
ax[0].plot(x_f1_cc, F1(x_f1_cc), '--', label="$f_1$ theory")
ax[0].set_xlabel('$x_2$')
ax[0].set_ylabel('$f_1(x_2)$')
ax[0].legend()
ax[1].plot(x_f2_cc, f2_cc, label='$f_2$ learned NN-CC')
ax[1].plot(x_f2_cc, F2(x_f2_cc), '--', label="$f_2$ theory")
ax[1].set_xlabel('$x_1$')
ax[1].set_ylabel('$f_2(x_1)$')
ax[1].legend()
plt.tight_layout()
plt.show()

