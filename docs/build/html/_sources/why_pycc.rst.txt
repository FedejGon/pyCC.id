========
Why pyCC
======== 

**pyCC.id** is a Python library for discovering interpretable, nonlinear dynamical systems from data. It is built on the concept of **Characteristic Curves (CCs)** and is designed to be highly customizable and user-friendly.

------------------------
Motivation: The Core Idea
------------------------

System identification (also known as equation discovery) is the process of finding the underlying governing equations of a system from observational data. 🔬 For many physical systems, the dynamics can be described by a set of first-order ordinary differential equations (ODEs):

.. math::

   \frac{d\mathbf{x}}{dt} = \mathbf{F}(\mathbf{x}, t)

Here, :math:`\mathbf{x}(t)` is the vector of the system's state variables (like position, velocity, etc.). The problem is that the function :math:`\mathbf{F}` can be incredibly complex and act like a "black box," making it difficult to gain physical insight.

The core philosophy of **pyCC.id** is to break down this complex function :math:`\mathbf{F}` into a combination of simpler, **interpretable building blocks**. This approach mirrors how a scientist or practitioner would construct a model: by considering different functions and parameters for modeling phenomena like stiffness, damping, or external forces.

We express this decomposition as:

.. math::

   \frac{d\mathbf{x}}{dt} = \mathbf{G}(\mathbf{x}, \mathbf{F}_{ext}(t); \{\mathbf{f}\}, \mathbf{a})

where:

* **Inputs**: :math:`\mathbf{x}` and :math:`\mathbf{F}_{ext}(t)`. These are the quantities you measure or control. :math:`\mathbf{x}` is the dynamical variable or **state** of the system; and :math:`\mathbf{F}_{ext}(t)` is a set of known, time-dependent **external forces**.

* **Model Components (Unknowns)**: The components to the right of the semicolon (`;`) are the unknowns the model seeks to find.

* :math:`\{\mathbf{f}\}`: A set of **unknown functions**, which we call the **Characteristic Curves**. In this approach, each function in this set depends on only a *single state variable* :math:`x_i`. This makes them interpretable (for example, one function could represent a nonlinear spring force, while another one an aerodynamic drag).

* :math:`\mathbf{a}`: A vector of **unknown scalar parameters**, such as mass, damping coefficients, or other physical constants.

* :math:`\mathbf{G}`: A proposed **model structure**. It defines the template that dictates how the building blocks (the functions :math:`\{\mathbf{f}\}` and parameters :math:`\mathbf{a}`) are combined with the state :math:`\mathbf{x}` to compute the system evolution. This structure can be an arbitrary user-defined function.

The goal of **pyCC** is to discover the optimal functions :math:`\{\mathbf{f}\}` and parameters :math:`\mathbf{a}` that best fit the observed data based on a predefined model structure :math:`\mathbf{G}`.

------------
Key Features
------------

* **Interpretable Models**: Decomposes complex dynamics into simpler, physically meaningful functions.
* **Flexible Function Parametrization**: Supports various techniques to model the characteristic curves, including:

    * Neural Networks (NN-CC) — Compatible with multicore CPUs and GPUs from both NVIDIA (CUDA) and Intel (XPU) architectures.
    * Polynomials (Poly-CC) — Using polynomial expansion basis functions for comparison.
    * Symbolic Regression (SymbR-CC) — Parallelized for multicore CPU execution, using the internal parallelization features of PySR.

* **Physics-Informed Discovery**: Incorporate known physical constraints, such as symmetries (e.g., even and odd functions) or conservation laws, to guide the discovery process and ensure robust, physically consistent models.
* **Built-in Simulator**: Includes a module for simulating higher-order and coupled ODEs, fully compatible with all identification methodologies.
* **User-Focused Design**: Offers an API that is both easy to use for standard problems and highly customizable for advanced research.
