import pycc
import numpy as np
import pandas as pd
#import pysindy as ps
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp

# --- Parameters ---
alpha  = 1.0
beta   = 0.2
delta  = 0.1
# --- Function to discover ---
def F1(x):
    return alpha*x + delta * x**2 + np.sin(x)
# --- Generating data to train models ---
x_span  = (-10, 10)
x_data  = np.linspace(*x_span, 1000)
F1_th  = F1(x_data)

plt.figure()
plt.plot(x_data,F1_th)
plt.xlabel("x")
plt.ylabel("F1(x)")
plt.show()


# --- Define DataFrame for training models ---
df = pd.DataFrame({
    'x': x_data,
    'F1_th':F1_th    
})
print(df.head())


################################ training model stage #############################
equations='f1(x)  = F1_th'

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
    plt.figure()
    plt.plot(x_f1_cc, f1_cc, label='f1_Poly learned')
    plt.plot(x_data,F1_th, '--', label="f1 theory")
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.legend()
    plt.show()
elif len(evals) == 4:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
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
    plt.show()

## The key is to loop through the evals list in steps of 2
#num_functions = len(evals) // 2
#if num_functions == 1:
#    x_f1_cc, f1_cc = evals
#elif num_functions == 2:
#    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
#elif num_functions == 3:
#    x_f1_cc, f1_cc, x_f2_cc, f2_cc, x_f3_cc, f3_cc = evals

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
          #### method SymbReg  ####
########################################
params_SymbReg = {
  'pysr': {
    'niterations': 100,
    'unary_operators': ['tanh','sin','cos'],
    'binary_operators': ['+','-','*'],
    'maxsize': 12,
    'populations':20,
    'model_selection': 'best', # 'best' , 'accuracy' , 'score'
    'verbosity': 0
  },
  'N_fit_points': 200,
  'max_iterations': 15,
}



models, evals , scalar_coefs = pycc.train(
    df=df,
    equations=equations,
    method='SymbReg',
    params=params_SymbReg
)


if len(evals) == 2:
    x_f1_cc, f1_cc = evals
    plt.figure()
    plt.plot(x_f1_cc, f1_cc, label='f1_SR learned')
    plt.plot(x_data,F1_th, '--', label="f1 theory")
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.legend()
    plt.show()
elif len(evals) == 4:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
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
    'epochs': 20000,
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

models, evals, obtained_coefs = pycc.train(
    df, 
    equations,
    method='NN', 
   # params=parameters_NN
)

if len(evals) == 2:
    x_f1_cc, f1_cc = evals
    plt.figure()
    plt.plot(x_f1_cc, f1_cc, label='f1_NN learned')
    plt.plot(x_data,F1_th, '--', label="f1 theory")
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.legend()
    plt.show()
elif len(evals) == 4:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
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



#equation='x_ddot + f1(x_dot) + f2(x) = F_ext'

