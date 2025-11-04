========
🎯 Why pyCC
======== 

**pyCC.id** is a Python library for discovering interpretable, nonlinear dynamical systems from data. It is built on the concept of **Characteristic Curves (CCs)** and is designed to be highly customizable and user-friendly.


The core problem **pyCC** aims to solve is a difficult choice often forced by other system identification tools:

1.  **Black-Box Models (e.g., Neural ODEs):**
    These methods are incredibly powerful and can fit complex dynamics with high accuracy. However, their internal workings are opaque, providing little to no physical insight. The result is a model that can *predict* but cannot *explain*.

2.  **Interpretable \'Library\' Methods (e.g., SINDy, SR):**

    * SINDy (Sparse Identification of Nonlinear Dynamics) is highly interpretable and computationally efficient. Its primary modeling assumption is that the dynamics can be sparsely represented in a **pre-defined library of candidate functions** (e.g., polynomials, trigonometric functions). This is highly effective if the true terms are in the library, but it can struggle if the underlying function is not (or cannot be well-approximated by) a sparse combination of these candidates. 
    * Pure Symbolic Regression (SR) is also highly interpretable and is generally more flexible than SINDy. It builds functions from a library of basic operators (e.g., +, \*, sin, cos). While this allows it to discover analytical expressions, applying it directly to a full, high-dimensional, and often noisy differential equation, can be computationally challenging. 
    * Additionally, both methods can be sensitive to hyperparameter tuning. Different settings can easily lead to different resulting models. 

**pyCC** is designed to fill the \'gap\' between these approaches. It is designed to find models that are both accurate and interpretable by not forcing this \'all-or-nothing\' choice.

------------------------
💡 The pyCC Approach: A Hybrid Framework
------------------------

The core idea of pyCC is to separate the known parts of a system equations from the unknown parts. It assumes the overall structure of the differential equations is known, but the specific forms of some nonlinear functions (the Characteristic Curves, or ``fi``) and possibly some parameters (``ai``) are not.

**pyCC** solves this by providing a flexible, multi-stage framework:

    * Discover the Dynamics (No Bias): Instead of guessing a library of functions, you can first use a powerful, non-biased approximator. The method='NN' uses a neural network to learn the numerical shape of these unknown ``fi`` functions directly from the data.

    * Enforce Physical Knowledge: pyCC allows you to add crucial domain knowledge. The constraints parameter (e.g., 'f1 odd', 'f2(0)=0') forces the model to obey physical constraints, dramatically reducing the search space and leading to more realistic solutions. Also, the possibility of adding multiple equations to the total loss functions offers a easy way to add physical quantities that are conserved throughout the motion (constants of motion).

    * Achieve Interpretability (Post-processing): **pyCC** library offers a easy form to find interpretable analytical expressions from the functions ``fi`` identified, (e.g. from the 'NN' fit).
    


------------------------
📝 Formalism
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
✨ Key Features
------------

* **Interpretable Models**: Decomposes complex dynamics into simpler, physically meaningful functions.
* **Flexible Function Parametrization**: Supports various techniques to model the characteristic curves, including:

    * Neural Networks (NN-CC) — Compatible with multicore CPUs and GPUs from both NVIDIA (CUDA) and Intel (XPU) architectures.
    * Polynomials (Poly-CC) — Using polynomial basis functions for comparison.
    * Symbolic Regression (SymbR-CC) — Parallelized for multicore CPU execution, using the internal parallelization features of PySR.

* **Physics-Informed Discovery**: Incorporate known physical constraints, such as symmetries (e.g., even and odd functions) or conservation laws, to guide the discovery process and ensure robust, physically consistent models.
* **Built-in Simulator**: Includes a module for simulating higher-order and coupled ODEs, fully compatible with all identification methodologies.
* **User-Focused Design**: Offers an API that is both easy to use for standard problems and highly customizable for advanced research.
