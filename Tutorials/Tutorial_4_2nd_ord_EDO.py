
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

# --- Parameters ---
alpha  = 1.0
beta   = 0.2
delta  = 0.1
F0     = 1.0
Omega  = 1.0
noise  = 0.0  # set >0 if you want noisy data
x0     = 0.0
v0     = 0.0

print(f"alpha={alpha}, beta={beta}, delta={delta}")
print(f"Omega={Omega}, F0={F0}, $x_0$={x0}, $v_0$={v0}")


# --- Force models ---
def Ff_coul(x_dot):
    """Simple Coulomb friction."""
    return 0.5 * np.tanh(500*x_dot)   # replace with your smooth_sign if needed
def F1(x_dot):
    return delta * x_dot + Ff_coul(x_dot)
def F2(x):
    return alpha * x + beta * x**3
def F_ext(t):
    return F0 * np.cos(Omega * t)

# --- ODE system ---
def S2_ode(t, y):
    x, x_dot = y
    x_ddot = F_ext(t) - F1(x_dot)*np.exp(x_dot*2) - F2(x)
    return [x_dot, x_ddot]


# --- Simulation ---
t_span  = (0, 20)
t_eval  = np.linspace(*t_span, 1000)
y0      = [x0, v0]  # initial conditions

sol = solve_ivp(S2_ode, t_span, y0, t_eval=t_eval, method='LSODA')
if sol.status != 0:
    raise RuntimeError(f"Integration failed: {sol.message}")

# --- Extract results ---
x_data     = sol.y[0] 
x_dot_data = sol.y[1]
time_data  = sol.t
F_ext_val  = F_ext(time_data)
# compute acceleration directly from system function
x_ddot_data = np.array([S2_ode(t, y)[1] for t, y in zip(time_data, sol.y.T)])

# --- Define DataFrame ---
df = pd.DataFrame({
    't': time_data,
    'x': x_data,
    'x_dot': x_dot_data,
    'x_ddot': x_ddot_data,
    'F_ext': F_ext_val
})

print(df.head())

#, method='BDF', rtol=1e-6, atol=1e-8, dense_output=True)
print(sol.status)   # 0 = success, 1 = reached event, -1 = failed
print(sol.message)
#, method='DOP853', rtol=1e-9, atol=1e-12)
#, method='Radau', rtol=1e-6, atol=1e-8
#, method='BDF'
#plt.plot(sol.t, sol.y[0])
#plt.xlabel("Time")
#plt.ylabel("x(t)")
#plt.title("Displacement")
#plt.grid(True)
#plt.show()

F1_th=F1(x_dot_data)
F2_th=F2(x_data)
#F1_th_noisy=F1(x_dot_data_noisy)
#F2_th_noisy=F2(x_data_noisy)


#plt.plot(x_dot_data, F1_th)
#plt.xlabel("x_dot")
#plt.ylabel("F1(x_dot)")
#plt.grid(True)
#plt.show()
#
#
#plt.plot(x_data, F2_th)
#plt.xlabel("x")
#plt.ylabel("F2(x)")
#plt.grid(True)
#plt.show()

#plt.figure()
#plt.plot(time_data,(x_ddot_data-F_ext_val+F1_th+F2_th)**2)
#plt.show()


#equation='x_ddot + f1(x_dot) + f2(x) = F_ext'


####################################3
##### automatic integrator from a defined equation string 
###########################3

# --- Force models ---
def Ff_coul(x_dot):
    """Simple Coulomb friction."""
    return 0.5 * np.tanh(500*x_dot)   # replace with your smooth_sign if needed
def F1(x_dot):
    return delta * x_dot + Ff_coul(x_dot)
def F2(x):
    return alpha * x + beta * x**3
def F_ext(t):
    return F0 * np.cos(Omega * t)

## --- ODE system ---
#def S2_ode(t, y):
#    x, x_dot = y
#    x_ddot = F_ext(t) - F1(x_dot) - F2(x)
#    return [x_dot, x_ddot]
#
## --- Simulation ---
#t_span  = (0, 20)
#t_eval  = np.linspace(*t_span, 1000)
#y0      = [x0, v0]  # initial conditions
#sol = solve_ivp(S2_ode, t_span, y0, t_eval=t_eval, method='LSODA')
#a1=2

equation1='x1_dot = x2'
equation2='x2_dot = F_ext - f1(x2) - f2(x1)'
equations = [equation1,equation2]
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
params_th = {
    "t_span": t_span,
    "y0": y0,  # x, x_dot, x_ddot
    "t_eval": t_eval,
    "method": "LSODA",
    'local_funcs': {'f1': lambda t: F1(t),'f2': lambda t: F2(t),'F_ext': lambda t: F_ext(t)},
 #   "local_funcs": {"f1": f1, "f2": f2, "F_ext": F_ext}
     "scalar_params": {'a1':2.0}
}
#equation = "x_ddot + f1(x_dot) + f2(x) - F_ext = 0"
sol,derivatives = pycc.simulate(equations,method="Theoretical", params=params_th)

# --- Extract results ---
time_data  = sol.t
x_data     = sol.y[0]
x_dot_data = sol.y[1]
F_ext_val  = F_ext(time_data)
x_ddot_data=derivatives[1]

#x_ddot_data = np.array([S2_ode(t, y)[1] for t, y in zip(time_data, sol.y.T)])

# --- Define DataFrame ---
df = pd.DataFrame({
    't': time_data,
    'x': x_data,
    'x_dot': x_dot_data,
    'x_ddot': x_ddot_data,
    'F_ext': F_ext_val
})
print(df.head())
print(sol.status)   # 0 = success, 1 = reached event, -1 = failed
print(sol.message)
#, method='DOP853', rtol=1e-9, atol=1e-12)
#, method='Radau', rtol=1e-6, atol=1e-8
#, method='BDF'
F1_th=F1(x_dot_data)
F2_th=F2(x_data)



x2=x_dot_data
x1_dot=x_dot_data

df = pd.DataFrame({
    't': time_data,
    'x': x_data,
    'x_dot': x_dot_data,
    'x_ddot': x_ddot_data,
    'F_ext': F_ext_val
})

x1=x_data
x2=x_dot_data

df = pd.DataFrame({
    't': time_data,
    'x1':x_data,
    'x2':x_dot_data,
    'x1_dot':x_dot_data,
    'x2_dot':x_ddot_data,
    'x': x_data,
    'x_dot': x_dot_data,
    'x_ddot': x_ddot_data,
    'F_ext': F_ext_val
})


#equation1='x_ddot + f1(x_dot) + f2(x)- F_ext  = 0'
#equation2='F_ext=(f1(x_dot))**2-f2(x)'

############## identification equation
eq1='x1_dot = x2'# *exp(a3-2)' #
eq2='x2_dot = F_ext - f1(x2) - f2(x1)'
equations = [eq1,eq2]

#df = pd.DataFrame({
#    't': time_data,
#    'x1':x_data,
#    'x2':x_dot_data,
#    'x1_dot':x_dot_data,
#    'x2_dot':x_ddot_data,
#    'x': x_data,
#    'x_dot': x_dot_data,
#    'x_ddot': x_ddot_data,
#    'F_ext': F_ext_val
#})


#equation1='x1_dot = x2'
#equation2='x2_dot = F_ext - f1(x_dot) - f2(x)'



#+ f1(x_dot) + f2(x)- F_ext  = 0'
#equation2='f2(x)=0'





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
  'max_iterations': 8,
}

models, evals , obtained_coefs = pycc.train(
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


############## Simulate SymbR code #############
params_SR_simul = {
    'models': models,
    'obtained_coefs': obtained_coefs,
    'local_funcs': {'F_ext': lambda t: F_ext(t)},
    't_span':t_span,
    'y0': y0,   # corresponds to equations order: first eq -> y0[0], second -> y0[1]
    't_eval': t_eval,
    'method': 'LSODA',  # solver for solve_ivp
    'atol': 1e-8,
    'rtol': 1e-6,
    'check_nan': True
}
sol,_  = pycc.simulate(equations, method='SymbR', params=params_SR_simul)
print("Integration success:", sol.success)

time_sim=sol.t
x_sim=sol.y[0]
x_dot_sim=sol.y[1]

# Plot solution
plt.figure()
plt.plot(time_sim, x_sim, label="x(t) simulated SR")
plt.plot(time_data, x_data, label="x(t) th")
plt.xlabel('t')
plt.ylabel('x(t)')
#plt.figure()
#plt.plot(sol.t, sol.y[1], label="x_dot(t) simulated NN(sym+SR)")
#plt.plot(time_data, x_dot_data, label="x_dot(t) th")
plt.legend()
plt.show()


########################################
          #### method Poly  ####
########################################
print("computing Poly")
params_poly={
  'scaling': True,
  'constraints': [
      #  {'constraint': 'f1(0)=0'},#,'penalty':1e1},
        {'constraint': 'f2(0)=0'},
        {'constraint': 'f1 odd'},
        {'constraint': 'f2 odd'}
    ],
   'learning_rate': 1e-3,
  'N_order': 20,
  'n_iter':4000,
  'eq_weights':[1.0,1.0]
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
plt.figure()
plt.plot(x_f1_cc, f1_cc, label='f1 learned')
plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
plt.xlabel('x_dot')
plt.ylabel('f1(x_dot)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2 learned')
plt.plot(x_data, F2_th, '--', label="f2 theory")
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend() 
#plt.figure()
#plt.plot(x_f3_cc, f3_cc, label='f3 learned')
##plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel('x_dot')
#plt.ylabel('f3(x_dot)')
#plt.legend() 
plt.show()






########################################
          #### method NN  ####
########################################
#equation='x_ddot + f1(x_dot) + a1*x + a2*x**3 - F_ext = 0'
#equation='x_ddot + a1*x_dot + a2*x + a3*x**3 - F_ext = 0'
constraints = [
    #{'constraint': 'f1(0)=0', 'penalty': 1e-2},
   #{'constraint': 'f2(0)=0', 'penalty': 1e-2},
   {'constraint': 'f1 odd'},# 'penalty': 1e-1, 'eval': 'array','Nval_array':100}, # penaly is optional
   {'constraint': 'f2 odd'},# 'penalty': 1e-1, 'eval': 'array','Nval_array':100}, # eval=data/array array is default
]
#constraints = [
#    {'constraint': 'f1(0)=0'},
#    {'constraint': 'f2(0)=0'},
#]
parameters_NN = {
    'neurons': 100,
    'layers':3,
    #'activation':'ReLu',
    'lr': 1e-3,
    'epochs': 5000,
    'error_threshold': 1e-6,
    'extrapolation': None,
    'weight_loss_param': 1e-3,
    # 'param_penalty_weight': 0.0,
    'constraints': constraints,
    'device': 'xpu',
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
elif num_functions == 2:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
elif num_functions == 3:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc, x_f3_cc, f3_cc = evals

#x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
plt.figure()
plt.plot(x_f1_cc, f1_cc, label='f1 learned')
plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
plt.xlabel('x_dot')
plt.ylabel('f1(x_dot)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2 learned')
plt.plot(x_data, F2_th, '--', label="f2 theory")
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend() 
#plt.figure()
#plt.plot(x_f3_cc, f3_cc, label='f3 learned')
##plt.plot(x_data, F2_th, '--', label="f2 theory")
#plt.xlabel('x_dot')
#plt.ylabel('f3(x_dot)')
#plt.legend() 
plt.show()
        
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

#####################################  simulate NN ##########################################
#pycc.save_trained_models('trained_models.pt', models, obtained_coefs)
#models, scalar_params = load_trained_models('trained_models.pt', device='cpu')

#equation1='x1_dot = x2'
#equation2='x2_dot = F_ext *(a1-1) - f1(x2) - f2(x1)'
#equations = [equation1,equation2]

params_NN_simul = {
    'models': models,
    'obtained_coefs': obtained_coefs,
    #'local_funcs': {'F_ext': lambda t: F0 * np.cos(Omega * t)},
    'local_funcs': {'F_ext': lambda t: F_ext(t)},
    #'local_funcs': {'F_ext'},
    't_span':t_span,
    'y0': y0,   # corresponds to equations order: first eq -> y0[0], second -> y0[1]
    't_eval': t_eval,
    'method': 'LSODA',  # solver for solve_ivp
    'atol': 1e-8,
    'rtol': 1e-6,
    'check_nan': True
}

sol = pycc.simulate(equations, method='NN', params=params_NN_simul)
print("Integration success:", sol.success)

time_sim=sol.t
x_sim=sol.y[0]
x_dot_sim=sol.y[1]

# Plot solution
plt.figure()
plt.plot(time_sim, x_sim, label="x(t) simulated NN(sym+SR)")
plt.plot(time_data, x_data, label="x(t) th")
plt.xlabel('t')
plt.ylabel('x(t)')
#plt.figure()
#plt.plot(sol.t, sol.y[1], label="x_dot(t) simulated NN(sym+SR)")
#plt.plot(time_data, x_dot_data, label="x_dot(t) th")
plt.legend()
plt.show()




###############################################################################################
print("now making POST-SR")
# Run symbolic regression post-processing
sr_results = pycc.process_evals_SymbR(
    evals,
    function_names=["f1", "f2"],  # must match training
    sr_params={"niterations": 100, "populations": 20},
    plot=True
)


# Print obtained symbolic expressions
print("\nFinal symbolic regression results:")
for fname, out in sr_results.items():
    print(f"{fname}(x) ≈ {out['expr']}")


# Option 2: Or manually create it with explicit scalar handling
f1 = sr_results["f1"]["func"]
f2 = sr_results["f2"]["func"]

def F_ext(t):
    return F0 * np.cos(Omega * t)

def ode_rhs(t, y):
    x, x_dot = y
    F_ext_val = F_ext(t)  # example forcing
    
    # Ensure scalar operations for ODE integration
    f1_val = f1(float(x_dot))  # Explicitly convert to scalar
    f2_val = f2(float(x))      # Explicitly convert to scalar
    
    # Ensure the results are scalars
    #if hasattr(f1_val, '__len__'):
    #    f1_val = float(f1_val[0])
    #else:
    #    f1_val = float(f1_val)
        
    #if hasattr(f2_val, '__len__'):
    #    f2_val = float(f2_val[0])
    #else:
    #    f2_val = float(f2_val)
    
    x_ddot = F_ext_val - f1_val - f2_val
    
    return [float(x_dot), float(x_ddot)]

# Now integration should work
y0 = [x0, v0]
#t_span = (0, 10)
#t_eval = np.linspace(0, 10, 500)

sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval)
time_sim=sol.t
x_sim=sol.y[0]
x_dot_sim=sol.y[1]

# Plot solution
plt.figure()
plt.plot(time_sim, x_sim, label="x(t) simulated NN(sym+SR)")
plt.plot(time_data, x_data, label="x(t) th")
plt.xlabel('t')
plt.ylabel('x(t)')
#plt.figure()
#plt.plot(sol.t, sol.y[1], label="x_dot(t) simulated NN(sym+SR)")
#plt.plot(time_data, x_dot_data, label="x_dot(t) th")
plt.legend()
plt.show()


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


