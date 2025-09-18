import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def simulate_th(equation_str, params, noise=0.0):
    """
    Forward simulator for 1st or 2nd order ODEs from string, returns full DataFrame with derivatives.
    """
    local_funcs = params.get("local_funcs", {})

    # Detect highest derivative
    if "x_ddot" in equation_str:
        order = 2
    elif "x_dot" in equation_str:
        order = 1
    else:
        order = 0

    # RHS function for integration
    def ode_rhs(t, y):
        x = y[0]
        x_dot = y[1] if order >= 2 else 0.0
        rhs_val = -f1_val(x_dot) - f2_val(x) + F_ext_val(t)
        return [x_dot, rhs_val] if order==2 else [rhs_val]

    # Extract callables
    f1_val = local_funcs.get("f1", lambda xd: 0.0)
    f2_val = local_funcs.get("f2", lambda x: 0.0)
    F_ext_val = local_funcs.get("F_ext", lambda t: 0.0)

    # Initial conditions
    y0 = params.get("y0", [0.0]*(order+1))
    t_span = params.get("t_span", (0,50))
    t_eval = params.get("t_eval", np.linspace(t_span[0], t_span[1], 5000))
    method = params.get("method", "LSODA")

    # Solve ODE
    sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, method=method)
    if sol.status != 0:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Extract data
    x_data = sol.y[0]
    x_dot_data = sol.y[1] if order >= 2 else np.gradient(x_data, sol.t)
    x_ddot_data = np.gradient(x_dot_data, sol.t)
    F_ext_values = F_ext_val(sol.t)

    # Add noise if requested
    if noise>0:
        x_data += np.random.normal(0, noise, size=x_data.shape)
        x_dot_data += np.random.normal(0, noise, size=x_dot_data.shape)
        x_ddot_data += np.random.normal(0, noise, size=x_ddot_data.shape)
        F_ext_values += np.random.normal(0, noise, size=F_ext_values.shape)

    # Build standard DataFrame
    df = pd.DataFrame({
        "t": sol.t,
        "x": x_data,
        "x_dot": x_dot_data,
        "x_ddot": x_ddot_data,
        "F_ext": F_ext_values
    })
    return df

