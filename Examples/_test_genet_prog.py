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
#equations = [
#    'x1_dot = x2',
#    'x2_dot = F_ext - f1(x2) - f2(x1)'
#]
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
F1_th = F1(x2_data)
F2_th = F2(x1_data)

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
#eqs = [
#    'x1_dot = x2',
#    'x2_dot = F_ext - f1(x2) - f2(x1)'
#]
eqs = [
    'x1_dot = f3(x2)',
    'x2_dot = F_ext - f1(x2)  - f2(x1)'
]
df = pd.DataFrame({
    #'t': time_data, # this can be added if eqs include t as an explicit variable
    'x1':x1_data,
    'x2':x2_data,
    'x1_dot':x1_dot_data,
    'x2_dot':x2_dot_data,
    'F_ext': F_ext_val
})


########################################
          #### method SparseR  ####
########################################
print("computing with Sparse Regression")


#params_sparse = {
#    'alpha': 1e-3,
#    'scaling': True,
#    'library': {
#        'default': [
#            # This list is used for any function NOT explicitly defined below
#            'x', 
#            'x**2', 
#            'x**3',
#            'tanh(500*x)',
#        ],
#        'f1': [
#            # A specific library just for function 'f1'
#            'sin(x)', 
#            'cos(x)',
#            'sin(2*x)',
#            'cos(2*x)'
#        ]
#        # 'f2' would use the 'default' library since it is not specified.
#    }
#}


# --- Parameter Definition ---
params_sparse = {
    'alpha': 1e-3,
    #'scaling': True,
    'library': {
        'default': [
            # Include polynomials up to order 5
            'polynomials(5)',
            
            # Include sine and cosine up to frequency 3
            #'sines(3)',
            #'cosines(3)',
            
            # Include custom functions for saturation or Gaussian-like effects
            #'tanh(x)',
            'tanh(500*x)',
            #'exp(-x**2)',
        ],
#        'f1': [
##            # A specific library just for function 'f1'
##            'sin(x)', 
##            'cos(x)',
##            'sin(2*x)',
##            'cos(2*x)'
#             'polynomials(6)',
#             'tanh(500*x)', 
#        ]
    },
    'tol':1e-6,
    'max_iter':2000
    
}


# Make sure your 'pycc' object can call this new method
models, evals, scalar_coefs = pycc.train(
    df=df,
    equations=equations,
    method='SparseR', 
    params=params_sparse
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
plt.plot(x2_data,F1_th, '--', label="f1 theory")
plt.xlabel('x_dot')
plt.ylabel('f1(x_dot)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2 learned')
plt.plot(x1_data, F2_th, '--', label="f2 theory")
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
          #### method SparseR  ####
########################################
print("computing with Sparse Regression")


params_gp = {
    # --- Population and Evolution Control ---
    'n_generations': 1500,      # (int) How many evolution cycles to run. More is better but slower.
    'population_size': 1500,    # (int) How many candidate solutions (individuals) in each generation.
    'crossover_rate': 0.7,     # (float, 0-1) Probability of combining two parents to create offspring.
    'mutation_rate': 0.5,      # (float, 0-1) Probability of randomly changing a part of an expression.
    'max_depth': 8,            # (int) The maximum depth of the initial random expression trees. Controls initial complexity.
    'tournament_size': 6,      # (int) How many individuals to select for a "tournament" to choose a parent. Higher values increase selection pressure.

    # --- Expression Building Blocks (The "Library") ---
    'library': {
        # (list of strings) Mathematical operations the GP can use.
        # Must be one of: 'add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'tanh'
        'functions': ['add', 'sub', 'mul', 'sin','tanh'],

        # (list of strings/numbers) Terminals or "leaves" of the expression tree.
        # 'x' is a special required placeholder for the function's input variable.
        # You can also include constant numbers.
        #'terminals': ['x', -1, 1, 2]
    },
    
    # --- Output Control ---
    'parsimony_coefficient':1e-2,
    'n_eval': 200              # (int) Number of points to use for generating the final plot data.
}
# Make sure your 'pycc' object can call this new method
models, evals, scalar_coefs = pycc.train(
    df=df,
    equations=equations,
    method='GP', 
    params=params_sparse
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
plt.plot(x2_data,F1_th, '--', label="f1 theory")
plt.xlabel('x_dot')
plt.ylabel('f1(x_dot)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2 learned')
plt.plot(x1_data, F2_th, '--', label="f2 theory")
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
