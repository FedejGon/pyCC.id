
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
F0     = 3.0
Omega  = 1.0
noise  = 0.0  # set >0 if you want noisy data
x0     = 0.5

print(f"alpha={alpha}, beta={beta}, delta={delta}")
print(f"Omega={Omega}, F0={F0}, $x_0$={x0}")

## Fext = f1(x) +f2(x) x'

# --- Force models ---
def Ff_coul(x_dot):
    """Simple Coulomb friction."""
    return 0.5 * np.sign(x_dot)   # replace with your smooth_sign if needed
def F1(x):
    return delta * x + 0.1* np.sin(3*x) #+ Ff_coul(x_dot)
def F2(x):
    return 1+np.abs(x) #alpha * x + beta * x**3+0.1
def F_ext(t):
    return F0 * np.cos(Omega * t)

# --- ODE system ---
def S1_ode(t, y):
    x = y
    x_dot = (F_ext(t) - F1(x))/ (F2(x))
    return x_dot


# --- Simulation ---
t_span  = (0, 50)
t_eval  = np.linspace(*t_span, 1000)
y0      = [x0]  # initial conditions

sol = solve_ivp(S1_ode, t_span, y0, t_eval=t_eval, method='LSODA')
if sol.status != 0:
    raise RuntimeError(f"Integration failed: {sol.message}")

# --- Extract results ---
x_data     = sol.y[0]
time_data  = sol.t
F_ext_val  = F_ext(time_data)
# compute acceleration directly from system function
#x_dot_data = np.array([S1_ode(t, y)[1] for t, y in zip(time_data, sol.y.T)])
x_dot_data = np.array([S1_ode(t, x) for t, x in zip(time_data, x_data)])


# --- Define DataFrame ---
df = pd.DataFrame({
    't': time_data,
    'x': x_data,
    'x_dot': x_dot_data,
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

F1_th=F1(x_data)
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











equation1='f1(x) + x_dot*f2(x) = F_ext'
#equation2='f2(x)=0'
#equation2=''
#equations = [equation1,equation2]
equations=equation1

########################################
          #### method Poly  ####
########################################
print("computing Poly")
params_poly={
  'scaling': True,
  'constraints': [
        #{'constraint': 'f1(0)=0'},#,'penalty':1e2},
        #{'constraint': 'f2(0)=0'},
        #{'constraint': 'f1 odd'},
        #{'constraint': 'f2 odd'}
    ],
  #'eq_weights':[1.0,0.0]
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
plt.plot(x_f1_cc, f1_cc, label='f1_Poly learned')
plt.plot(x_data,F1_th, '+', label="f1 theory")
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2_Poly learned')
plt.plot(x_data, F2_th, '+', label="f2 theory")
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
          #### method SymbReg  ####
########################################
params_SymbReg = {
  'pysr': {
    'niterations': 100,
    'unary_operators': ['sin','cos','tanh'],
    'binary_operators': ['+','-','*'],
    'maxsize': 12,
    'populations':10,
    'model_selection': 'accuracy', # 'best' , 'accuracy' , 'score'
    'verbosity': 0
  },
  'N_fit_points': 200,
  'max_iterations': 25,
}



models, evals , scalar_coefs = pycc.train(
    df=df,
    equations=equations,
    method='SymbReg',
    params=params_SymbReg
)


if len(evals) == 2:
    x_f1_cc, f1_cc = evals
elif len(evals) == 4:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals

# then your plotting code:
plt.figure()
plt.plot(x_f1_cc, f1_cc, label='f1_SR learned')
plt.plot(x_data,F1_th, '+', label="f1 theory")
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2_SR learned')
plt.plot(x_data, F2_th, '+', label="f2 theory")
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
    'lr': 1e-4,
    'epochs': 10000,
    'error_threshold': 1e-6,
    'extrapolation': None,
    'weight_loss_param': 1e1,
    # 'param_penalty_weight': 0.0,
    #'constraints': constraints,
    #'eq_weights': [1.0, 0.0]
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

plt.figure()
plt.plot(x_f1_cc, f1_cc, label='f1_NN learned')
plt.plot(x_data,F1_th, '+', label="f1 theory")
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2_NN learned')
plt.plot(x_data, F2_th, '+', label="f2 theory")
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


