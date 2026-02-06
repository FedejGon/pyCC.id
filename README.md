<div align="center">

# pyCC.id: A Python package for nonlinear equation discovery based on characteristic curves


[![GitHub repository](https://img.shields.io/badge/GitHub-FedejGon/pyCC.id-blue?style=flat-square&logo=github)](https://github.com/FedejGon/pyCC.id)

**pyCC.id** is a Python library for discovering interpretable, nonlinear dynamical systems from data. It is built on the concept of **Characteristic Curves (CCs)** and is designed to be highly customizable and user-friendly.

**pyCC** is a user-friendly and highly-customizable Python library for data-driven equation discovery, designed to bridge the gap between *black-box* and *white-box* modeling paradigms, while facilitating practical applications in science and engineering.

| **colab demo** | **Forums** | **Paper** |
|:---:|:---:|:---:|
|[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/drive/136FvEwMsxLayhimgtI4Jx_IWR8l-dy-s)|[![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/FedejGon/pyCC.id/discussions)|[![Paper](https://img.shields.io/badge/arXiv-2305.01582-b31b1b)](https://arxiv.org/)||

</div>

---


## 🎯 Core Idea

System identification (also known as equation discovery) is the process of finding the underlying governing equations of a system from observational data.  For many physical systems, the dynamics can be described by a set of first-order ordinary differential equations (ODEs):

$$
\frac{d\mathbf{x}}{dt} = \mathbf{F}(\mathbf{x}, t)
$$

Here, $\mathbf{x}(t)$ is the vector of the system's state variables (like position, velocity, etc.). The problem is that the function $\mathbf{F}$ can be incredibly complex and act like a "black box," making it difficult to gain physical insight.


The core philosophy of **pyCC.id** is to break down this complex function $\mathbf{F}$ into a combination of simpler, **interpretable building blocks**. This approach mirrors how a scientist or practitioner would construct a model: by considering different functions and parameters for modeling phenomena like stiffness, damping, or external forces.

We express this decomposition as:

$$
\frac{d\mathbf{x}}{dt} = \mathbf{G}(\mathbf{x}, \mathbf{F}_{ext}(t); \\{\mathbf{f}\\}, \mathbf{a})
$$

where:

* **$\mathbf{x}$** and **$\mathbf{F}_{ext}(t)$** are the **inputs** to the model: $\mathbf{x}$ is the dynamical variable or **state** of the system; and $\mathbf{F}_{ext}(t)$ is a set of known, time-dependent **external forces**. These are the quantities you measure or control.

* The semicolon **`;`** separates the variables of the system from the components of the model you are trying to find. To the left are the inputs; to the right are the unknowns that define the model.

* **$\\{\mathbf{f}\\}$** is a set of **unknown functions**, which we call the **Characteristic Curves**. In this approach, each function in this set depends on only a *single state variable* $x_i$. This makes them interpretable (for example, one function could represent a nonlinear spring force, while another one an aerodynamic drag).

* **$\mathbf{a}$** is a vector of **unknown scalar parameters**, such as mass, damping coefficients, or other physical constants.

* **$\mathbf{G}$** represents a proposed **model structure**. It defines the template that dictates how the building blocks (the functions $\\{\mathbf{f}\\}$ and parameters $\mathbf{a}$) are combined with the state $\mathbf{x}$ to compute the system evolution. This structure can be an arbitrary user-defined function.

The goal of **pyCC** is to discover the optimal functions $\\{\mathbf{f}\\}$ and parameters $\mathbf{a}$ that best fit the observed data based on a predefined model structure $\mathbf{G}$.


## 🔬 Why pyCC?

**i) Identifiability**
When inferring dynamical equations from real experiments (often with finite sampling or noisy data), multiple distinct mathematical models can fit the observations with comparable accuracy. This leads to **ambiguity in model selection**. **pyCC** addresses this by injecting **prior physical knowledge** into the discovery process via a structural 'skeleton'. When the hypothesized model structure possesses uniqueness properties, **pyCC** provides a formal framework to assess whether the proposed equation is consistent with the data.

**ii) Physical Consistency**
To define physically motivated model structures, we use the formalism of **Characteristic Curves (CCs)**. This approach decomposes high-dimensional dynamics into modular, **univariate functions**. In this view, each CC represents a constitutive relation of an independent physical element (e.g., a specific spring or damper). This assures **physical consistency**: the learned model is not just a curve fit, but a collection of distinct physical mechanisms.

**iii) Interpretability**
The use of CCs allows the practitioner to 'visualize' the model simply by plotting the univariate curves.
* *Traditional approach:* "Find the coefficients $k$ and $c$ assuming linear dynamics."
* *pyCC approach:* "Find the **shapes** of the stiffness and damping curves."

If the stiffness curve looks like a parabola, we know the system is nonlinear. This visual insight allows for qualitative discovery before quantitative fitting.

**iv) Modularity, Universality, and Transparency**
Since **pyCC** prioritizes the **shape** of the constitutive relations over their specific model coefficients, the parametric form (e.g., polynomial vs. exponential) does not need to be postulated *a priori*.
We can parameterize the CCs using **universal approximators**, such as Neural Networks (the **NN-CC** approach).
* **Universality:** The model can adapt to any continuous shape regardless of complexity, provided sufficient model capacity.
* **Transparency:** While NNs are often regarded as "black boxes," within **pyCC** they are restricted to learning **univariate** functions. A "black box" with a single input and single output is effectively transparent: it is simply a curve that can be plotted and visually inspected to interpret the underlying physics.



## 🔬 Application Example: Second-Order System

Consider identifying a second-order system with a velocity-dependent friction force and external driving force. The practitioner starts by hypothesizing the skeleton:

$$
\ddot{x} + f_1(\dot{x}) + f_2(x) = F_{ext}(t)
$$

This equation implies two CCs: a **damping force** $f_1(\dot{x})$ and a **restoring force** $f_2(x)$.

<div align="center">
<img src="docs/source/_static/Fig2_model_veloc.png" width="70%" alt="Neural Network architecture for a second-order system">

*Figure 2: The architecture for a second-order system. Two independent neural networks ($\text{NN}_1$ and $\text{NN}_2$) approximate the unknown CCs. $\text{NN}_1$ sees only velocity, and $\text{NN}_2$ sees only position.*
</div>

**Why this architecture matters:**
Crucially, this architecture enforces uniqueness and physical consistency. Even if the training data contains complex transient behaviors, the model **cannot** learn spurious cross-terms (like $x\dot{x}$) because no single module has access to both variables simultaneously.


---

## 📖 Example: Nonlinear Oscillator

Let's consider a second-order nonlinear differential equation:

$$
\ddot{x} + \delta\dot{x} + \mu\tanh(500\dot{x}) + \alpha x + \beta x^3 = F_{ext}(t)
$$

where $F_{ext}(t) = A\cos(\omega t)$. The term $\tanh(500\dot{x})$ is a smooth approximation of the signum function, $\text{sign}(\dot{x})$, often used to model Coulomb friction.

For compatibility with higher-order systems, we recommend rewritting the system into a set of first-order equations. By defining the state variables $x_1 = x$ and $x_2 = \dot{x}$, the system becomes:

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - \delta x_2 - \mu\tanh(500x_2) - \alpha x_1 - \beta x_1^3
\end{cases}
$$

After simulating this system, the input data that will be used for identification is defined by **$x_i$**, **$\dot{x}_{i}$** and **$F_{ext}$**.

### Define an Identification Strategy
With **pyCC.id**, you can face the identification problem in several ways:

#### (i) Functional Approach
Here, we assume the structure of the equation but leave key components as unknown functions to be discovered from data.

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - f_1(x_2) - f_2(x_1)
\end{cases}
$$

The goal is to find the shapes of the characteristic curves $f_1$ and $f_2$. These functions can be parameterized using neural networks, polynomials, or other methods.


#### (ii) Parametric Approach
If you have a strong hypothesis about the functional forms, you can identify the unknown parameters directly.

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - a_1x_2 - a_2\tanh(a_3x_2) - a_4x_1 - a_5 x_1^3
\end{cases}
$$

The goal is to find the optimal values for the parameters $\\{a_i\\}$ using nonlinear iterative algorithms.

#### (iii) Hybrid Approach
This approach combines the functional and parametric methods. You can assume known forms for some parts of the equation while leaving other parts as unknown functions.

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - f_1(x_2) - a_4 x_1 - a_5 x_1^3
\end{cases}
$$

Here, we identify the function $f_1(\dot{x})$ and the parameters $a_4$ and $a_5$ simultaneously.


## 📥  Installation with pip (Recommended)

### Installation on CPU and Nvidia GPUs
Some features in PyCC include using the Symbolic Regression (pySR) package. Thus we recommend installing this package first. To install both packages use:  
```bash
pip install pycc.id
```

### Installation on Intel XPUs
To run this library on Intel XPUs, you must install the *intel-extension-for-pytorch* package compatible with your operative system. Please refer to the official instructions at https://pytorch-extension.intel.com/installation.

Below are examples for installing version v2.8.10+xpu.
For Linux/WSL2 OS; first, install PyTorch and Intel extension packages: 
```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
python -m pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
For Windows OS; use instead:  
```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.8.10+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

Final step: once the environment is set up, install the remaining packages from the PyCC library:
```bash

pip install pycc.id
```




### Installation for developers (from source)
Download or clone the repository and install with:
```bash
pip install -e .
```

## 🚀 Usage

```bash
# Import the package into your Python environment
import pycc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This example shows:
# 1) how to simulate a stick-slip second order system using pycc.simulate()
# 2) how to train the NN-CC method to identify the model [pycc.train()]
# 3) how to simulate the identified model [pycc.simulate()]

##############################################
# 1) how to simulate a stick-slip second order system using pycc.simulate()
# 1a) define parameters and functions
alpha=1.0;beta=0.2;delta=0.1;Omega=1.0;
x0=0.0;v0=0.0; y0=[x0,v0] # initial conditions
t_span=(0, 20); t_eval=np.linspace(*t_span, 1000)
def F1_th(x_dot):
    return delta * x_dot + 0.5 * np.tanh(500*x_dot)
def F2_th(x):
    return alpha * x + beta * x**3
def F_ext(t):
    return np.cos(Omega * t)

# 1b) define equation
eqs_th = ['x1_dot = x2',
          'x2_dot = F_ext - f1(x2) - f2(x1)']

# 1c) define simulation parameters
params_th = {
    't_span': t_span,
    'y0': y0,  
    't_eval': t_eval,
    'method': 'LSODA',
    'local_funcs': {'f1': lambda t: F1_th(t),'f2': lambda t: F2_th(t),'F_ext': lambda t: F_ext(t)}
}
# 1d) integrate forward the theoretical equation
sol,derivatives = pycc.simulate(eqs_th,method="Theoretical", params=params_th)

# 1e) extract data from theoretical solution
time_data  = sol.t
x1_data     = sol.y[0]
x2_data = sol.y[1]
x1_dot_data=derivatives[0]
x2_dot_data=derivatives[1]
F_ext_val  = F_ext(time_data)

# define database for training
df = pd.DataFrame({
    'x1':x1_data,
    'x2':x2_data,
    'x1_dot':x1_dot_data,
    'x2_dot':x2_dot_data,
    'F_ext': F_ext_val
})

##############################################
# 2) how to train the NN-CC method to identify the model [pycc.train()]
# 2a) propose equations to use for identification (fi functions and ai parameters).
eqs = [
     'x1_dot = x2', #*exp(a1-2.0)',
     'x2_dot = F_ext - f1(x2) - f2(x1)'
]
# 2b) define constraints (optional)  
constraints = [ # adding prior known information
   {'constraint': 'f2(0)=0'},
   {'constraint': 'f1 odd'},
   {'constraint': 'f2 odd'},
]
# 2c) define training parameters (optional)
params_NN = {
    'neurons': 100,
    'layers':3,
    'lr': 1e-4,
    'epochs': 2000,
    'error_threshold': 1e-6,
    'extrapolation': None,
    'device':'cpu',
    'weight_loss_param': 1e-3,
    'constraints': constraints,
}
# 2d) train/fit/identify the model
models, evals, obtained_coefs = pycc.train(df, eqs,method='NN', params=params_NN)

# plotting obtained functions f1 and f2
x_f1_cc, f1_cc, x_f2_cc, f2_cc = evals
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(x_f1_cc, f1_cc, label='$f_1$ learned NN-CC')
ax[0].plot(x_f1_cc, F1_th(x_f1_cc), '--', label="$f_1$ theory")
ax[0].set_xlabel('$x_2$')
ax[0].set_ylabel('$f_1(x_2)$')
ax[0].legend()
ax[1].plot(x_f2_cc, f2_cc, label='$f_2$ learned NN-CC')
ax[1].plot(x_f2_cc, F2_th(x_f2_cc), '--', label="$f_2$ theory")
ax[1].set_xlabel('$x_1$')
ax[1].set_ylabel('$f_2(x_1)$')
ax[1].legend()
plt.tight_layout()
plt.show()

# Print learned parameters (if any)
if obtained_coefs:
    print("\nLearned scalar parameters:")
    for name, val in obtained_coefs.items():
        print(f"{name} = {val.item():.4f}")


##############################################
# 3) how to simulate the identified model [pycc.simulate()]

### Simulation using the NN models
print("simulation with NN simul")
# 3a) define simulation parameters
params_NN_simul = {
    'models': models,
    'obtained_coefs': obtained_coefs,
    'local_funcs': {'F_ext': lambda t: F_ext(t)},
    't_span':t_span,
    'y0': y0,   
    't_eval': t_eval,
    'method': 'LSODA',  # solve_ivp
    'atol': 1e-8,
    'rtol': 1e-6,
    'check_nan': True
}
# 3b) integrate identified equations
sol,_ = pycc.simulate(eqs, method='NN', params=params_NN_simul)
print("Integration success:", sol.success)

time_sim=sol.t
x1_sim=sol.y[0]
x2_sim=sol.y[1]

# Identified vs theoretical solution
plt.figure()
plt.plot(time_sim, x1_sim, label="x(t) simulated NN(sym+SR)")
plt.plot(time_data, x1_data, label="x(t) th")
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()

```


> ⏳ **Initial import delay** : The first time you run *import pycc*, it may take around 3 minutes to set up dependencies. This is a one-time process; after that, imports will be nearly instantaneous.


## 📚  Tutorials
**First time? We recommend starting with our Google Colab Notebook** [![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/drive/136FvEwMsxLayhimgtI4Jx_IWR8l-dy-s)!

Additionally, various tutorials and examples are available in the *Tutorials* folder. You can download or copy these files to your local machine or cluster, and execute them directly, for example:
```bash
python Tutorial1.py
```


---

## ✨ Key Features

* **Interpretable Models**: Decomposes complex dynamics into simpler, physically meaningful functions.
* **Flexible Function Parametrization**: Supports various techniques to model the characteristic curves, including:
    * Neural Networks (NN-CC) — Compatible with multicore CPUs and GPUs from both NVIDIA (CUDA) and Intel (XPU) architectures. GPU acceleration on Intel devices is enabled through the intel_extension_for_pytorch.
    * Polynomials (Poly-CC) — Using polynomial expansion basis functions for comparison.
    * Symbolic Regression (SymbR-CC)  —  Parallelized for multicore CPU execution, using the internal parallelization features of PySR.
* **Physics-Informed Discovery**: Incorporate known physical constraints, such as symmetries (e.g., even and odd functions) or conservation laws, to guide the discovery process and ensure robust, physically consistent models.
* **Built-in Simulator**: Includes a module for simulating higher-order and coupled ODEs, fully compatible with all identification methodologies.
* **User-Focused Design**: Offers an API that is both easy to use for standard problems and highly customizable for advanced research.

---






## 🏛️ How to cite this package

General reference to this package:
**Gonzalez2026code**



In case of using NN-CC method, please cite:
  - Gonzalez, F. J. "[Integrating prior knowledge in equation discovery: Interpretable symmetry-informed neural networks and symbolic regression via characteristic curves]( 	
https://doi.org/10.48550/arXiv.2601.21720)." arXiv preprint arXiv:2601.21720 (2026).
  - Gonzalez, F. J. and Lara, L. P. "[Interpretable neural network system identification method for two families of second-order systems based on characteristic curves](https://doi.org/10.1007/s11071-025-11744-6)." Nonlinear Dyn. 113, 33063–33086 (2025).

In case of using Poly-CC method, please cite:
  - Gonzalez, F.J. "[System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems](https://doi.org/10.1007/s11071-024-09890-4)." Nonlinear Dyn. 112, 16167–16197 (2024).
  - Gonzalez, F.J. "[Determination of the characteristic curves of a nonlinear first order system from fourier analysis](https://doi.org/10.1038/s41598-023-29151-5)." Sci. Rep., vol. 13, 1955, (2023).
  
In case of using post-SR and/or SymbReg-CC methods, please cite:
  - Gonzalez, F. J. "[Integrating prior knowledge in equation discovery: Interpretable symmetry-informed neural networks and symbolic regression via characteristic curves]( 	
https://doi.org/10.48550/arXiv.2601.21720)." arXiv preprint arXiv:2601.21720 (2026).
  - Cranmer, M. "[Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl](https://doi.org/10.48550/arXiv.2305.01582)." arXiv preprint arXiv:2305.01582 (2023).

## Citation BibTex


```bibtex
@article{Gonzalez2026,
  title={Integrating prior knowledge in equation discovery: Interpretable symmetry-informed neural networks and symbolic regression via characteristic curves}, 
  author={Federico J. Gonzalez},
  year={2026},
  eprint={2601.21720},
  archivePrefix={arXiv},
  primaryClass={nlin.CD},
  url={https://arxiv.org/abs/2601.21720}, 
}

@article{Gonzalez2025nody,
  title = {{Interpretable neural network system identification method for two families of second-order systems based on characteristic curves}},
  volume = {113},
  ISSN = {1573-269X},
  DOI = {10.1007/s11071-025-11744-6},
  number = {24},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  author = {Gonzalez,  Federico J. and Lara,  Luis P.},
  year = {2025},
  month = sep,
  pages = {33063–33086}
}

@article{Gonzalez2024,
  title = {System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems},
  author = {{F. J. Gonzalez}},
  volume = {112},
  issn = {1573-269X},
  doi = {10.1007/s11071-024-09890-4},
  number = {18},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  year = {2024},
  month = jul,
  pages = {16167–16197}
}

@article{Gonzalez2023,
  title     = {Determination of the characteristic curves of a nonlinear first order system from Fourier analysis},
  author    = {Gonzalez, Federico J.},
  journal   = {Sci. Rep.},
  publisher = {Springer Science and Business Media LLC},
  volume    =  13,
  number    =  1,
  pages     = {1955},
  month     =  feb,
  year      =  2023,
  doi =   {10.1038/s41598-023-29151-5},
}

@article{Cranmer2023PySR,
  title={Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl},
  author={Miles Cranmer},
  journal={arXiv preprint arXiv:2305.01582},      
  year={2023},
  eprint={2305.01582},
  url={https://arxiv.org/abs/2305.01582},
}
```


### 🤝 We are open to collaborations and adding new possible features.
Please share your [![Ideas](https://img.shields.io/badge/ideas-github-informational)](https://github.com/FedejGon/pyCC.id/discussions/categories/ideas) or reach out for a possible collaboration to:
 - Federico J. Gonzalez: fgonzalez@ifir-conicet.gov.ar

🐞 For **Issues** or **bugs**, add new [![Issues](https://img.shields.io/badge/issue-github-informational)](https://github.com/FedejGon/pyCC.id/issues).   
