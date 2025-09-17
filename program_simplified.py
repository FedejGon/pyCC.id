
# to do:
#1) compatibility with gpu cuda and gpu intel


#import pycc
import pycc

import numpy as np
import pandas as pd

#from pycc_train import train_nn_models
#import pycc_train
#import pysindy as ps
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter



#parameters stick slip
m=1.0 # kg
cval=0.1 # Ns/m (viscous damping coefficient)
kval=1.0 # N/m (stiffness)
F0=2 # N (forcing amplitude)
Omega=0.3 # 0.3 and 0.15 rad/s (forcing frequency)
x0=0.1 # m (initial displacement)
v0=0.1 # m/s (initial velocity)
mu_N = 0.5 #0.5

#parameters duffing
m=1.0
alpha=-1.0
beta=1.0
delta=0.3
x0=0.5
v0=0.5
F0=0.5
omega=1.2


# Condiciones iniciales y tiempo de simulación
#friction definition Dietrich-Ruina
#Ff=0.5 # N
#a=0.07
#b=0.09
##c=0.022
#Vf=0.003 # m/s
#epsilon=1e-6 # m/s
Tsimul=40
Nsimul=500
Tval=2*Tsimul
Nval=2*Nsimul
noise=0.0


t_span = (0, Tsimul)  # Intervalo de tiempo
t_simul = np.linspace(*t_span, Nsimul)  # Puntos de evaluación
t_span_val = (0, Tval)  # Intervalo de tiempo
t_val = np.linspace(*t_span_val, Nval)  # Puntos de evaluación

alpha=-1.0
beta=1.0
delta=0.3
x0=0.5
v0=-0.5
F0=0.5
Omega=1.2
print(f"alpha={alpha}, beta={beta}, delta={delta}")
print(f"Omega={Omega}, F0={F0}, $x_0$={x0}, $v_0$={v0}")


SNR_dB=[np.inf] #   20
#SNR_dB=20

y0 = [x0, v0]  # [x(0), x'(0)]
#y0_val = [x0, v0]
#y0 = [0.5, 0.0, 0.0]  # [x(0), x'(0), t(0)]

# Hyperparameters for NN
learning_rate = 1e-3
epochs_max = 20000
neurons=100
error_threshold = 1e-8
f1_symmetry='odd'
f2_symmetry='odd'
lambda_penalty = 1e-1  # You can adjust this weight if needed
lambda_penalty_symm = 1e-1
apply_restriction=False

def smooth_sign(x, alpha=500):
    return np.tanh(alpha * x)
def Ff_dr(x_dot):
    abs_v = np.abs(x_dot) + epsilon
    #abs_v = np.clip(np.abs(x_dot), 1e-6, 1e2)  # Prevent too small or too big
    abs_v = np.maximum(np.abs(x_dot), 1e-10)  # Prevent division by zero
    term1 = a * np.log(abs_v / Vf)
    term2 = b * np.log(c + Vf / abs_v)
    #return (Ff + term1 + term2) * np.sign(x_dot)
    return (Ff + term1 + term2) * smooth_sign(x_dot)
def Ff_dieterich_ruina(x_dot):
  abs_v = np.maximum(np.abs(x_dot), epsilon)
  return (p0 + a * np.log(abs_v / Vf)) * smooth_sign(x_dot)
def Ff_striebeck(x_dot):
  abs_v = np.maximum(np.abs(x_dot), epsilon)
  b_exp = 1.0  # smoothness exponent
  return (p0 + a * np.exp(-(abs_v / Vf)**b_exp)) * smooth_sign(x_dot)
def Ff_coul_tanh(x_dot):
  alpha2=50
  return mu_N * np.tanh(alpha2 * x_dot)
def Ff_coul_tanh_power(x_dot):
  alpha2=500
  npow=3.0
  return mu_N * np.tanh(alpha2 * x_dot**npow)

def Ff_coul(x_dot):
 #   return mu_N * np.sign(x_dot)
    return mu_N * smooth_sign(x_dot)
def F1(x_dot):
    return delta * x_dot + Ff_coul(x_dot) #cval* x_dot + Ff_coul(x_dot) # + 0.0005 * x_dot**2 #+ Ff_coul(x_dot) #r(x_dot) Ff_coul Ff_dr
def F2(x):
    return alpha*x+beta*x**3 #kval*x
def F_ext(t):
    return F0*np.cos(Omega*t)
def S1_stick_slip(t,y):
    x, x_dot = y  # y=[x, x']
    x_ddot = (F_ext(t) - F1(x_dot) - F2(x))/m
    return [x_dot, x_ddot]


def Ff_coul_anderson2009(x_dot,F_ext):
    #abs_x_dot = np.abs(x_dot)
    #if abs_x_dot > 0: #1e-8:
    #    return mu_N * np.sign(x_dot)
    #    #return mu_N * smooth_sign(x_dot)
    #else:
    #    return np.minimum(np.abs(F_ext), mu_N) * np.sign(F_ext)
    abs_x_dot = np.abs(x_dot)
    abs_fext = np.abs(F_ext)
    sign_fext = np.sign(F_ext)

    # If scalar: return scalar
    if np.isscalar(x_dot):
        if abs_x_dot > 1e-8:
            return mu_N * np.sign(x_dot) # smooth_sign(x_dot)  #
        else:
            return min(abs_fext, mu_N) * sign_fext

    # If array: return array
    result = np.where(
        abs_x_dot > 1e-8,
        mu_N * smooth_sign(x_dot),
        np.minimum(abs_fext, mu_N) * sign_fext
    )
    return result
#   return mu_N * np.sign(x_dot)
#   return mu_N * smooth_sign(x_dot)
#    return mu_N * np.sign(x_dot) if |x_dot>0? else min(|Fext|,mu_N)*np.sign(Fext)
def F1_anderson2009(x_dot,F_ext):
    return c* x_dot + Ff_coul_anderson2009(x_dot,F_ext) #r(x_dot) Ff_coul Ff_dr
def F2_anderson2009(x):
    return kval*x/m
def S1_stick_slip_anderson2009(t,y):
    x, x_dot = y  # y=[x, x']
    fext_val = F_ext(t)
#    friction = Ff_coul_anderson2009(x_dot, fext_val)
    x_ddot = (fext_val - F1_anderson2009(x_dot,fext_val) - F2_anderson2009(x)) / m
    return [x_dot, x_ddot]

#  1 x'' + 0.1 * x' + 0.5 sign(x') + k * x  = F_ext(t)

# EDO: m x'' + c * x' + Ff(x') + k * x  = F_ext(t)
# wn=sqrt(k/m)
# c=zeta*2*sqrt(k*m)
# F_ext= F0 cos(Omega * t)
#Ff={Ff+a*ln[(|x'|+epsilon)/Vf]+b*ln[c+Vf/(|x'|+epsilon)]}sgn(x')

#def van_der_pol_with_time_F_discontinuous(t,y):
#    x, x_dot = y  # x, x', and time
#    if x > 1:  # Introducing a discontinuity when x > 1
##        x_ddot = -x  # Ignore the Van der Pol term and set x_ddot to just -x
#        f = 2*mu * (2 - x**2)  # Ignore the Van der Pol term and set x_ddot to just -x
#    else:
#        f = mu * (1 - x**2)  # Original Van der Pol term when x <= 1
#    x_ddot = f * x_dot - x
#    return [x_dot, x_ddot]


# Generar datos de la EDO con solve_ivp
#sol = solve_ivp(van_der_pol_with_time_F, t_span, y0, t_eval=t_eval)

#sol = solve_ivp(S1_stick_slip, t_span, y0, t_eval=t_simul)

#def stick_event(t, y):
#    return y[1]  # Detect when velocity crosses zero
#stick_event.terminal = False
#stick_event.direction = 0  # Detect all zero crossings
#sol = solve_ivp(S1_stick_slip, t_span, y0, t_eval=t_simul,
#                events=stick_event, method='Radau')

#sol = solve_ivp(S1_stick_slip, t_span, y0, t_eval=t_simul)
sol = solve_ivp(S1_stick_slip, t_span, y0, t_eval=t_simul,method='LSODA') #LSODA


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


x_data = sol.y[0]+np.random.normal(0,0.1)*noise      # Posición
x_dot_data = sol.y[1] #+np.random.normal(0,0.01)   # Velocidad
time_data = sol.t      # Time (x2)



# Add noise to x_data for a given SNR (in dB)
#SNR_dB = 0  # desired signal-to-noise ratio in decibels
#signal_power = np.mean(x_data**2)
#noise_power = signal_power / (10**(SNR_dB / 10))
#noise_std = np.sqrt(noise_power)
#x_data_noisy = x_data + np.random.normal(0, noise_std, size=x_data.shape)
#x_noise_substraction = x_data_noisy - x_data
#signal_power = np.mean(x_data**2)
#noise_power = np.mean(x_noise_substraction**2)
#snr_measured = 10 * np.log10(signal_power / noise_power)
#print(f"Desired SNR: {SNR_dB} dB")
#print(f"Measured SNR: {snr_measured:.2f} dB")

# Add noise to x_data for a given SNR (in dB)

if np.isinf(SNR_dB):
    print("Running with SNR = ∞ dB (no noise)")
    print("noise=",noise)
    F_ext_val = F_ext(time_data)+np.random.normal(0,0.1)*noise
    noise_percentage=0.0
    noise_percentage_th=0.0
else:
    print(f"Running with SNR = {SNR_dB:.2f} dB")
    # Add noise based on current SNR_dB

    SNR_dB = 2.0 #4  # desired signal-to-noise ratio in decibels
    Fext_signal_power = np.mean(F_ext(time_data)**2)
    noise_power = Fext_signal_power / (10**(SNR_dB / 10))
    noise_std = np.sqrt(noise_power)
    F_ext_val_noisy = F_ext(time_data) + np.random.normal(0, noise_std, size=time_data.shape)
    Fext_noise_substraction = F_ext_val_noisy - F_ext(time_data)
    signal_power = np.mean(F_ext(time_data)**2)
    noise_power = np.mean(Fext_noise_substraction**2)
    snr_measured = 10 * np.log10(signal_power / noise_power)

    # Compute noise percentage relative to RMS signal
    signal_rms = np.sqrt(signal_power)
    noise_percentage_th=100*10**(-SNR_dB / 20.0)
    noise_percentage = 100 * (noise_std / signal_rms)
    #F_fr=Ff_dr(x_dot_data)
    #F1_th=F1(x_dot_data)
    #F2_th=F2(x_data)
    #F_ext_val = F_ext(time_data)+noise*np.random.normal(0,0.5)
    print(f"Desired SNR in Fext: {SNR_dB} dB")
    print(f"Measured SNR in Fext: {snr_measured:.2f} dB")
    print(f"Noise percentage in Fext: {noise_percentage:.2f}%")
    print(f"Noise percentage in Fext (theoretical): {noise_percentage_th:.2f}%")
    # --- now apply a Savitzky–Golay filter ---
    # choose an odd window length and a small polynomial order
    window_length = 51    # must be odd, e.g. 5, 11, 51, …
    polyorder     = 3     # < window_length

    F_ext_filtered = savgol_filter(
        F_ext_val_noisy,
        window_length=window_length,
        polyorder=polyorder,
        mode='interp'       # avoids edge artifacts
    )

    # measure the SNR *after* filtering (optional)
    noise_after = F_ext_filtered - F_ext(time_data)
    snr_after   = 10 * np.log10(
        np.mean(F_ext(time_data)**2) / np.mean(noise_after**2)
    )
    print(f"SNR after SG filter: {snr_after:.1f} dB")

#    plt.figure(figsize=(6, 4))
#    plt.plot(time_data, F_ext(time_data),         label='Fext (true)')
#    plt.plot(time_data, F_ext_val_noisy,          label='Fext + noise', alpha=0.7)
#    plt.plot(time_data, F_ext_filtered,           label='SG-filtered', linewidth=2)
#    plt.xlabel('Time')
#    plt.ylabel('Fₑₓₜ')
#    plt.title('Original vs Noisy vs SG-Filtered Forcing')
#    plt.legend()
#    plt.tight_layout()
#    plt.show()


    #F_fr=Ff_dr(x_dot_data)
    #F1_th=F1(x_dot_data)
    #F2_th=F2(x_data)
    F_ext_val = F_ext(time_data)+np.random.normal(0,0.1)*noise
    F_ext_val = F_ext_filtered
    F_ext_val = F_ext_val_noisy

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


#F1_th=F1_anderson2009(x_dot_data,F_ext_val)
#F2_th=F2_anderson2009(x_data)


#x_ddot_data = (F_ext_val - F1_th - F2_th) / m
#x_ddot_data = np.gradient(x_dot_data, sol.t)  # Aceleración (derivada numérica)
x_ddot_data = np.array([S1_stick_slip(t, y)[1] for t, y in zip(sol.t, sol.y.T)])
#x_ddot_data = np.array([S1_stick_slip_anderson2009(t, y)[1] for t, y in zip(sol.t, sol.y.T)])


#plt.figure()
#plt.plot(time_data,(x_ddot_data-F_ext_val+F1_th+F2_th)**2)
#plt.show()



x_dot_data=x_dot_data*1

#model1, model2 = pycc.train(
#    t_simul, x_data, x_dot_data, x_ddot_data, F_ext_val,
#    neurons=100, learning_rate=1e-3, epochs_max=5000,
#    lambda_penalty=1.0, apply_restriction=True
#)


df = pd.DataFrame({
    'x': x_data,
    'x_dot': x_dot_data,
    'x_ddot': x_ddot_data,
    'F_ext': F_ext_val
})


#equation='x_ddot + f1(x_dot) + f2(x) = F_ext'

equation1='x_ddot + f1(x_dot) + f2(x) - F_ext * exp(2*a1) = 0'
equation2='f1(x_dot)=1'
equations = [equation1,equation2]

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
    'constraints': constraints,
    'eq_weights': [1.0, 0.0]
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



########################################
          #### method SymbReg  ####
########################################
params_SymbReg = {
  'pysr': {
    'niterations': 80,
    'unary_operators': ['tanh'],
    'binary_operators': ['+','-','*'],
    'maxsize': 25,
    'verbosity': 0
  },
  'N_fit_points': 200
}

models, evals , scalar_coefs = pycc.train(
    df=df,
    equation=equation1,
    method='SymbReg',
    params=params_SymbReg
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



########################################
          #### method Poly  ####
########################################
print("computing Poly")


equation2='f1(x_dot)=1'
equations=[equation1,equation2]
params_poly={
  'scaling': True,
  #'constraints': [
  #      #{'constraint': 'f1(0)=0'},
  #      {'constraint': 'f2(0)=0'},
  #      #{'constraint': 'f1 odd'},
  #      {'constraint': 'f2 odd'}
  #  ],
  'eq_weights':[1.0,0.0]
}

models, evals , scalar_coefs = pycc.train(
    df=df,
    equation=equations,
    method='Poly',
    params=params_poly
)







if len(evals) == 2:
    x_f1_cc, f1_cc = evals
elif len(evals) == 4:
    x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals

# then your plotting code:
x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
plt.figure()
plt.plot(x_f1_cc, f1_cc, label='f1 Poly')
plt.plot(x_dot_data,F1_th, '--', label="f1 theory")
plt.xlabel('x_dot')
plt.ylabel('f1(x_dot)')
plt.legend()
plt.figure()
plt.plot(x_f2_cc, f2_cc, label='f2 Poly')
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


