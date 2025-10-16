<div align="center">

# pyCC :  High-Performance System Identification using Characteristic Curves

[![GitHub repository](https://img.shields.io/badge/GitHub-FedejGon/pyCC.id-blue?style=flat-square&logo=github)](https://github.com/FedejGon/pyCC.id)

**pyCC.id** is a Python library for discovering interpretable, nonlinear dynamical systems from data. It is built on the concept of **Characteristic Curves (CCs)** and is designed to be highly customizable and user-friendly.


| **colab demo** | **Forums** | **Paper** | 
|:---:|:---:|:---:|
|[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/drive/136FvEwMsxLayhimgtI4Jx_IWR8l-dy-s)|[![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/FedejGon/pyCC.id/discussions)|[![Paper](https://img.shields.io/badge/arXiv-2305.01582-b31b1b)](https://arxiv.org/)||

</div>

---

## 🎯 Core Idea

System identification (also known as equation discovery) is the process of finding the underlying governing equations of a system from observational data.  🔬 For many physical systems, the dynamics can be described by a set of first-order ordinary differential equations (ODEs):

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

* **$\mathbf{x}$** and **$\mathbf{F}_{ext}(t)$** are the **inputs** to the model: $\mathbf{x}$ is the dynamical variable or **state** of the system; and $\mathbf{F}_{ext}(t)$ is a set of known, time-dependent **external forces** or inputs. These are the quantities you measure or control.

* The semicolon **`;`** separates the variables of the system from the components of the model you are trying to find. To the left are the inputs; to the right are the unknowns that define the model.

* **$\\{\mathbf{f}\\}$** is a set of **unknown functions**, which we call the **Characteristic Curves**. The key insight of our method is that each function $f_j$ in this set typically depends on only a *single state variable* $x_i$. This makes them interpretable—for example, one function could represent a nonlinear spring force (**$f(x_{position})$**), while another represents aerodynamic drag (**$f(x_{velocity})$**).

* **$\mathbf{a}$** is a vector of **unknown scalar parameters**, such as mass, damping coefficients, or other physical constants.

* **$\mathbf{G}$** represents the **model structure** you propose. It's the template that dictates how the building blocks—the functions $\\{\mathbf{f}\\}$ and parameters $\mathbf{a}$—are combined with the state $\mathbf{x}$ to compute the system's evolution. This structure is not limited to a simple sum and can be an arbitrary user-defined function.

The goal of **pyCC** is to discover the optimal functions $\\{\mathbf{f}\\}$ and parameters $\mathbf{a}$ that make your proposed model structure $\mathbf{G}$ best fit the observed data. By finding the forms of the functions in $\{\mathbf{f}\}$ and the values of the parameters in $\mathbf{a}$, **pyCC** helps you discover a transparent and physically meaningful model of your system.

---

## ✨ Key Features

* **Interpretable Models**: Decomposes complex dynamics into simpler, physically meaningful functions.
* **Flexible Function Parametrization**: Supports various techniques to model the characteristic curves, including:
    * Neural Networks (NN-CC)
    * Polynomials (Poly-CC)
    * Symbolic Regression (SymbR-CC)
* **Built-in Simulator**: Includes a module for simulating higher-order and coupled ODEs, which is fully compatible with all identification methodologies.
* **User-Focused Design**: Aims for an API that is both easy to use for standard problems and highly customizable for advanced research.

---

## 📖 Example: Identifying a Nonlinear Oscillator

Let's consider a second-order nonlinear differential equation:

$$
\ddot{x} + \delta\dot{x} + \mu\tanh(500\dot{x}) + \alpha x + \beta x^3 = F_{ext}(t)
$$

where $F_{ext}(t) = A\cos(\omega t)$. The term $\tanh(500\dot{x})$ is a smooth approximation of the signum function, $\text{sign}(\dot{x})$, often used to model Coulomb friction.


* ### Step 1: Generating data

For compatibility with higher-order systems, we recommend rewritting the system into a set of first-order equations. By defining the state variables $x_1 = x$ and $x_2 = \dot{x}$, the system becomes:

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - \delta x_2 - \mu\tanh(500x_2) - \alpha x_1 - \beta x_1^3
\end{cases}
$$

After simulating this system, the input data that will be used for identification is defined  **$\\{x_1, x_2, \dot{x}_1, \dot{x}_2, F_{ext}\\}$** (or equivalently, **$\\{x, \dot{x}, \ddot{x}, F_{ext}\\}$**).

* ### Step 2: Define an Identification Strategy

With **pyCC.id**, you can face the identification problem in several ways:

    * #### (i) Functional Approach
Here, we assume the structure of the equation but leave key components as unknown functions to be discovered from data.

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - f_1(x_2) - f_2(x_1)
\end{cases}
$$

The goal is to find the shapes of the characteristic curves $f_1$ and $f_2$. These functions can be parameterized using neural networks, polynomials, or other methods.


    * #### (ii) Parametric Approach
If you have a strong hypothesis about the functional forms, you can identify the unknown parameters directly.

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - a_1x_2 - a_2\tanh(a_3x_2) - a_4x_1 - a_5x_1^3
\end{cases}
$$

The goal is to find the optimal values for the parameters $\\{a_i\\}$ using nonlinear iterative algorithms.

    * #### (iii) Hybrid Approach
This approach combines the functional and parametric methods. You can assume known forms for some parts of the equation while leaving other parts as unknown functions.

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = F_{ext}(t) - f_1(x_2) - a_1x_1 - a_2x_1^3
\end{cases}
$$

Here, we identify the function $f_1(\dot{x})$ and the parameters $a_1$ and $a_2$ simultaneously.


## 📥  Installation with pip (Recommended) 

### Installation for users
Some features in PyCC include using the Symbolic Regression (pySR) package. Thus we recommend installing this package first. To install both packages use:  
```bash
pip install pysr
pip install pycc.id
```

### Installation for developers (from source)
Download or clone the repository and install with:
```bash
pip install -e .
```

## 🚀 Usage
Import the package into your Python environment:
```bash
import pycc
```
> ⏱️ **Initial import delay** : the very first time you run *import pycc* after a fresh installation, the package automatically sets up dependencies. This process may take ∼3 minutes. Subsequent imports will be fast.

## 📚  Tutorials
**First time? We recommend starting with our Google Colab Notebook** [![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/drive/136FvEwMsxLayhimgtI4Jx_IWR8l-dy-s)!

Additionally, various tutorials and examples are available in the *Tutorials* folder. You can download or copy these files to your local machine or a cluster, and execute them directly, for example:
```bash
python Tutorial1.py
```



---






## 🖋️ How to cite this package

General reference to this package:
**Gonzalez2025code**



In case of using NN-CC method, additionally cite:
  - Gonzalez, F. J. and Lara, L. P. "[Interpretable neural network system identification method for two families of second-order systems based on characteristic curves](https://doi.org/10.1007/s11071-025-11744-6)." Nonlinear Dyn. (2025)
  
In case of using Poly-CC method, additionally cite: 
  - Gonzalez, F.J. "[Determination of the characteristic curves of a nonlinear first order system from fourier analysis](https://doi.org/10.1038/s41598-023-29151-5)." Sci. Rep., vol. 13, 1955, (2023).
  - Gonzalez, F.J. "[System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems](https://doi.org/10.1007/s11071-024-09890-4)." Nonlinear Dyn. 112, 16167–16197 (2024). 

In case of using post-SR and/or SymbReg-CC methods, additionally cite:
  - Cranmer, M. "[Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl](https://doi.org/10.48550/arXiv.2305.01582)." arXiv preprint arXiv:2305.01582 (2023).


## Citation BibTex


```bibtex
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

@article{Gonzalez2024,
  title = {System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems},
  author = {{F. J. Gonzalez}},
  volume = {112},
  issn = {1573-269X},
  url = {},
  doi = {10.1007/s11071-024-09890-4},
  number = {18},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  year = {2024},
  month = jul,
  pages = {16167–16197}
}

@article{Gonzalez2025nody,
  title = {Interpretable neural network system identification method for two families of second-order systems based on characteristic curves},
  author = {Gonzalez,  Federico J. and  Lara, Luis P. },
  doi = {10.1007/s11071-025-11744-6},
  volume = {},
  issn = {},
  number = {},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  year = {2025},
  month = sep,
  pages = {}
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


