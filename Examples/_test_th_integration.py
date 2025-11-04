import numpy as np
import pycc
import matplotlib.pyplot as plt

# Define the known theoretical functions
def f1_theory(x_dot):
    return 0.1 * x_dot

def f2_theory(x):
    return 0.5 * x + 0.2 * x**3

def F_ext_theory(t):
    return 1.0 * np.cos(0.5 * t)

eqs = [
    'x1_dot = x2',
    'x2_dot = F_ext - a1*f1(x2) - f2(x1)'
]

sim_params_th = {
    'local_funcs': {
        'f1': f1_theory,
        'f2': f2_theory,
        'F_ext': F_ext_theory
    },
    'scalar_params': {'a1': 2.0},
    't_span': (0, 100),
    'y0': [1.0, 0.0],  # Initial state [x1(0), x2(0)]
    't_eval': np.linspace(0, 100, 1000)
}

sol, derivs = pycc.simulate(eqs, method='Theoretical', params=sim_params_th)


# Extract results
time_sim = sol.t
x1_sim = sol.y[0]  # State variable x1(t)
x2_sim = sol.y[1]  # State variable x2(t)

# Derivatives are also returned
x1_dot_sim = derivs[0] # x1_dot(t)
x2_dot_sim = derivs[1] # x2_dot(t)

# Plot solution (assuming 'time_data', 'x_data' exist for comparison)
plt.figure()
plt.plot(time_sim, x1_sim, label="x(t) simulated")
#plt.plot(time_data, x_data, 'r--', label="x(t) data")
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()
