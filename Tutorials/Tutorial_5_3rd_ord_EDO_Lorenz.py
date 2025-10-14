
# to do:
#1) compatibility with gpu cuda and gpu intel
import pycc
import numpy as np
import pandas as pd
#import pysindy as ps
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# --- Parameters --- Chaotic solutions
sigma  = 10.0  # a1
rho   = 28.0  # a2
beta  = 8.0/3.0 # a3~2.666666666
x0     = 0.5
y0     = 0.8
z0     = -0.3
initial_state = [x0, y0, z0]


# --- ODE system ---
def S3_ode_lorenz(t, state):
    """
    Defines the Lorenz system of differential equations.
    
    Args:
        t (float): The current time.
        state (list): A list or array containing the current state [x, y, z].
    
    Returns:
        list: The derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]



# --- Simulation ---
t_span  = (0, 20)
t_eval  = np.linspace(*t_span, 3000)

#sol = solve_ivp(S3_ode_lorenz, t_span, initial_state,dense_output=True, t_eval=t_eval)#, method='LSODA')
sol = solve_ivp(S3_ode_lorenz, t_span, initial_state, t_eval=t_eval,method='LSODA')
if sol.status != 0:
    raise RuntimeError(f"Integration failed: {sol.message}")


# --- Plot the results ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Lorenz Attractor')
plt.show()
#, method='BDF', rtol=1e-6, atol=1e-8, dense_output=True)
print(sol.status)   # 0 = success, 1 = reached event, -1 = failed
print(sol.message)
#, method='DOP853', rtol=1e-9, atol=1e-12)
#, method='Radau', rtol=1e-6, atol=1e-8
#, method='BDF'

# --- Extract results ---
x_data = sol.y[0]
y_data = sol.y[1]
z_data = sol.y[2]
time_data = sol.t

# --- Compute derivatives ---
# Use a list comprehension to calculate dx/dt, dy/dt, and dz/dt at each time point
derivatives = [S3_ode_lorenz(t, [x, y, z]) for t, x, y, z in zip(time_data, x_data, y_data, z_data)]
dxdt_data = [d[0] for d in derivatives]
dydt_data = [d[1] for d in derivatives]
dzdt_data = [d[2] for d in derivatives]

dxdt_data, dydt_data, dzdt_data=S3_ode_lorenz(time_data, [x_data, y_data, z_data])


# --- Define DataFrame ---
df = pd.DataFrame({
    't': time_data,
    'x': x_data,
    'y': y_data,
    'z': z_data,
    'dxdt': dxdt_data,
    'dydt': dydt_data,
    'dzdt': dzdt_data
})

print(df.head())




####################################3
##### automatic integrator from a defined equation string 
###########################3

# Functions defined in main code
#def Ff_coul(x_dot):
#    """Simple Coulomb friction."""
#    return 0.5 * np.sign(x_dot)   # replace with your smooth_sign if needed
#def f1(x_dot):
#    return delta * x_dot #+ Ff_coul(x_dot)
#def f2(x):
#    return alpha * x + beta * x**3
#def F_ext(t):
#    return F0 * np.cos(Omega * t)
## Parameters for simulation
#params_th = {
#    "t_span": (0, 50),
#    "y0": [0.0, 0.0],  # x, x_dot, x_ddot
#    "t_eval": np.linspace(0, 50, 5000),
#    "method": "LSODA",
#    "local_funcs": {"f1": f1, "f2": f2, "F_ext": F_ext}
#}
#equation = "x_ddot + f1(x_dot) + f2(x) - F_ext = 0"
#df = pycc.simulate(equation,method="Theoretical", params=params_th)
#print(df.head())

# --- Define functions as strings ---
def_eq1 = 'dxdt = a1 * (y - x)'
#def_eq1 = 'dxdt = f1(x) * (y - x)'
def_eq2 = 'dydt = x * (a2 - z) - y'
def_eq3 = 'dzdt = x * y - a3 * z'

# Combine the equations into a single list
equations = [def_eq1, def_eq2, def_eq3]

#equation1='x_ddot + f1(x_dot) + f2(x)- F_ext  = 0'
#equation2='f2(x)=0'




########################################
          #### method NN  ####
########################################
#equation='x_ddot + f1(x_dot) + a1*x + a2*x**3 - F_ext = 0'
#equation='x_ddot + a1*x_dot + a2*x + a3*x**3 - F_ext = 0'
constraints = [
    {'constraint': 'f1(0)=0', 'penalty': 1e-2},
   #{'constraint': 'f2(0)=0', 'penalty': 1e-2},
    {'constraint': 'f1 odd', 'penalty': 1e-1, 'eval': 'array','Nval_array':100}, # penaly is optional
    {'constraint': 'f2 odd', 'penalty': 1e-1, 'eval': 'array','Nval_array':100}, # eval=data/array array is default
]
#constraints = [
#    {'constraint': 'f1(0)=0'},
#    {'constraint': 'f2(0)=0'},
#]
parameters_NN = {
    'neurons': 100,
    'lr': 1e-3,
    'epochs': 5000,
    'error_threshold': 1e-6,
    'extrapolation': None,
    'weight_loss_param': 1e-3,
    # 'param_penalty_weight': 0.0,
    #'constraints': constraints,
    #'eq_weights': [1.0, 1.0]
}
#models, evals, obtained_coefs = pycc.train(
#    df=df,
#    equation=equation1,
#    method='NN',
#    params=parameters_NN
#)
#equation1='x_ddot + f1(x_dot) + f2(x) - F_ext = 0'

models, evals, obtained_coefs = pycc.train(df, equations,method='NN', params=parameters_NN)


# The key is to loop through the evals list in steps of 2
num_functions = len(evals) // 2
if num_functions == 1:
    x_f1_cc, f1_cc = evals
    plt.figure()
    plt.plot(x_f1_cc, f1_cc, label='f1 learned')
    #plt.plot(x_data,np.ones(x_data)*sigma, '--', label="f1 theory")
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.legend()
    plt.figure()
    plt.show()
elif num_functions == 2:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
elif num_functions == 3:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc, x_f3_cc, f3_cc = evals

#x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
#plt.figure()
#plt.plot(x_f1_cc, f1_cc, label='f1 learned')
#plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
#plt.xlabel('x_dot')
#plt.ylabel('f1(x_dot)')
#plt.legend()
#plt.figure()
#plt.plot(x_f2_cc, f2_cc, label='f2 learned')
#plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel('x')
#plt.ylabel('f2(x)')
#plt.legend() 
#plt.figure()
#plt.plot(x_f3_cc, f3_cc, label='f3 learned')
##plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel('x_dot')
#plt.ylabel('f3(x_dot)')
#plt.legend() 
#plt.show()
        
#    plt.figure()
#    plt.plot(x_f1_cc, f1_cc, label='f1 learned')
#    plt.plot(x_dot_data, F1_th, '--', label='f1 theory')
#    plt.xlabel('x_dot')
#    plt.ylabel('f1(x_dot)')
#    plt.legend()
#    plt.show()


#    plt.figure()
#    plt.plot(x_f1_cc, f1_cc, label='f1 learned')
#    plt.plot(x_dot_data, F1_th, '--', label='f1 theory')
#    plt.xlabel('x_dot')
#    plt.ylabel('f1(x_dot)')
#    plt.legend()
#
#    plt.figure()
#    plt.plot(x_f2_cc, f2_cc, label='f2 learned')
#    plt.plot(x_data, F2_th, '--', label='f2 theory')
#    plt.xlabel('x')
#    plt.ylabel('f2(x)')
#    plt.legend()
#    plt.show()
    
# Print learned parameters (if any)
if obtained_coefs:
    print("\nLearned scalar parameters:")
    for name, val in obtained_coefs.items():
        print(f"{name} = {val.item():.4f}")


#models, evals = pycc.train(df, equation, neurons=50, lr=1e-2, epochs=2000)

## Plot learned functions
#x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
#plt.figure()
#plt.plot(x_f1_cc, f1_cc, label='f1 learned')
#plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
#plt.xlabel('x_dot')
#plt.ylabel('f1(x_dot)')
#plt.legend()
#plt.figure()
#plt.plot(x_f2_cc, f2_cc, label='f2 learned')
#plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel('x')
#plt.ylabel('f2(x)')
#plt.legend() 
#plt.show()




########################################
          #### method Poly  ####
########################################
print("computing Poly")
params_poly={
  'scaling': False,
  'constraints': [
      #  {'constraint': 'f1(0)=0'},#,'penalty':1e1},
        #{'constraint': 'f2(0)=0'},
        #{'constraint': 'f1 odd'},
        #{'constraint': 'f2 odd'}
    ],
  #'eq_weights':[1.0,1.0]
}
models, evals , scalar_coefs = pycc.train(
    df=df,
    equations=equations,
    method='Poly', #method='Poly_linear',
    params=params_poly
)
if len(evals) == 2:
    x_f1_cc, f1_cc = evals
elif len(evals) == 4:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals

# The key is to loop through the evals list in steps of 2
num_functions = len(evals) // 2
if num_functions == 1:
    x_f1_cc, f1_cc = evals
elif num_functions == 2:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
elif num_functions == 3:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc, x_f3_cc, f3_cc = evals

#x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
#plt.figure()
#plt.plot(x_f1_cc, f1_cc, label='f1 learned')
#plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
#plt.xlabel('x_dot')
#plt.ylabel('f1(x_dot)')
#plt.legend()
#plt.figure()
#plt.plot(x_f2_cc, f2_cc, label='f2 learned')
#plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel('x')
#plt.ylabel('f2(x)')
#plt.legend() 
#plt.figure()
#plt.plot(x_f3_cc, f3_cc, label='f3 learned')
##plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel('x_dot')
#plt.ylabel('f3(x_dot)')
#plt.legend() 
#plt.show()



########################################
          #### method SymbR  ####
########################################
params_SymbR = {
  'pysr': {
    'niterations': 100,
    'unary_operators': ['tanh'],
    'binary_operators': ['+','-','*'],
    'maxsize': 12,
    'populations':10,
   'model_selection': 'best', # 'best' , 'accuracy' , 'score'
    'verbosity': 0
  },
  'N_fit_points': 200,
  'max_iterations': 15,
}



models, evals , scalar_coefs = pycc.train(
    df=df,
    equations=equations,
    method='SymbR',
    params=params_SymbR
)


if len(evals) == 2:
    x_f1_cc, f1_cc = evals
elif len(evals) == 4:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals

# then your plotting code:
x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
plt.figure()
plt.plot(x_f1_cc, f1_cc, label='f1 SR')
plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
plt.xlabel('x_dot')
plt.ylabel('f1(x_dot)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2 SR')
plt.plot(x_data, F2_th, '--', label="f2 theory")
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend()
plt.show()









################################################
################ plotting to do #####################

#x_dot_cc, f1_cc, x_cc, f2_cc  = pycc.print_cc(df, equation, models)
#plt.figure()
#plt.plot(x_dot_cc, f1_cc, label="f1 learned")
#plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
#plt.xlabel("x_dot")
#plt.ylabel("f1")
#plt.legend()
#plt.grid(True)
#plt.show()
#plt.figure()
#plt.plot(x_cc, f2_cc, label="f2 learned")
#plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel("x")
#plt.ylabel("f2")
#plt.legend()
#plt.grid(True)
#plt.show()


