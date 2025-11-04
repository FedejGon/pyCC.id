---
title: "pyCC.id: A Python package for nonlinear equation discovery based on characteristic curves"
tags:
  - Python
  - nonlinear dynamics
  - equation discovery
  - system identification
  - characteristic curves
authors:
  - name: Federico J. Gonzalez
    orcid: 0000-0003-2026-4129
    affiliation: 1
affiliations:
  - name: Institute of Physics Rosario. Blvd. 27 de Febrero 210 Bis, S2000EKF Rosario, Santa Fe, Argentina.
    index: 1
date: 5 November 2025
bibliography: paper.bib
---

# Summary

Data-driven system identification often forces a choice between two extremes. On one side, **black-box models** (like Neural ODEs) can fit complex dynamics with high accuracy, but their internal workings are opaque, providing prediction without physical explanation. On the other side, **interpretable 'library' methods** (like SINDy [@Brunton2016] or pure Symbolic Regression [@Cranmer2023PySR]) find simple equations, but their success depends on the true dynamics being sparsely represented in a pre-defined library of candidate functions or the correct selection of hyperparameters. A small modification of input hyperparameters typically yield to different identified models. 



`pyCC.id` fills the gap between these approaches with a hybrid "grey-box" framework. Its core assumption is that the dynamics, $\mathbf{F}(\mathbf{x}, t)$, can be decomposed into a user-defined model structure $\mathbf{G}$ built from simpler, interpretable components: a set of unknown one-dimensional functions $\{\mathbf{f}\}$ (the characteristic curves, e.g., $f_i$(x)) and a set of scalar parameters $\mathbf{a}$.


Instead of guessing a library for the full dynamics, a user can first employ a powerful, non-biased approximator (e.g., a neural network) to learn the numerical shape of the unknown 1D functions `fi`. Crucially, physical knowledge (like symmetries or conservation laws) can be enforced as constraints. In a second stage, symbolic regression can be applied to this learned function to find a simple, interpretable analytical expression.
 
This framework builds upon a methodology first developed for first-order systems using a polynomial approach [@Gonzalez2023; @Gonzalez2024] and later extended to second-order systems using neural networks [@Gonzalez2025nody; @Gonzalez2025arxiv]. The pyCC.id package generalizes this work, providing a flexible tool applicable to a wider range of higher-order systems.
 

The package provides a practical tool for domain experts in engineering, physics, and biology to translate complex time-series data into simple, interpretable governing equations.

# Formalism

For many physical systems, the dynamics are described by a set of first-order ordinary differential equations (ODEs):

$$
\frac{d\mathbf{x}}{dt} = \mathbf{F}(\mathbf{x}, t)
$$

Here, $\mathbf{x}(t)$ is the system state vector, but the function $\mathbf{F}$ is often a complex "black box," making it difficult to gain physical insight. The core philosophy of `pyCC.id` is to decompose this complex function $\mathbf{F}$ into a combination of simpler, interpretable building blocks. We express this decomposition as:

$$
\frac{d\mathbf{x}}{dt} = \mathbf{G}(\mathbf{x}, \mathbf{F}_{ext}(t); \{\mathbf{f}\}, \mathbf{a})
$$

Where $\mathbf{x}$ and $\mathbf{F}_{ext}(t)$ are the measured inputs (system state and external forces). The goal is to find the model components: $\{\mathbf{f}\}$, a set of unknown 1D **Characteristic Curves** (e.g., nonlinear stiffness or damping functions), and $\mathbf{a}$, a vector of unknown scalar parameters. The user proposes the **model structure** $\mathbf{G}$, which defines how these components are combined. `pyCC.id` is a Python package that discovers the optimal functions $\{\mathbf{f}\}$ and parameters $\mathbf{a}$ that best fit the observed data.



# Features

`pyCC.id` is designed to be a flexible and high-performance tool for researchers and practitioners. Its key features include:

* **Interpretable Models**: Decomposes complex, high-dimensional dynamics into a set of simple, 1D characteristic curves that often have direct physical meaning (e.g., stiffness, damping).
* **Flexible Function Parametrization**: Supports multiple backends for modeling the characteristic curves, allowing users to choose the right tool for their problem:
    * **Neural Networks (NN-CC)**: Uses `PyTorch` [@Paszke2019pytorch] for high-performance, non-biased function approximation.
    * **Polynomials (Poly-CC)**: Provides a simple baseline using polynomial basis functions.
    * **Symbolic Regression (SymbR-CC)**: Uses `PySR` [@Cranmer2023PySR] to discover analytical expressions directly.
* **Physics-Informed Discovery**: Allows users to inject domain knowledge as constraints during training (e.g., `'f1 odd'`, `'f2(0)=0'`) or by defining conserved quantities in the loss function. This leads to more robust and physically consistent models.
* **Hardware Acceleration**: Natively supports multicore CPUs and GPUs from both NVIDIA (`cuda`) and Intel (`xpu` via `intel-extension-for-pytorch`) for accelerating neural network training.
* **Built-in Simulator**: Includes a versatile ODE simulator (`pycc.simulate`) compatible with all identified model types for validation and analysis.
* **Comprehensive Documentation**: Provides a Google Colab notebook for a quick start, as well as a full gallery of tutorials and examples in the repository.




# Acknowledgements

This work was partially supported by CONICET (Consejo Nacional de Investigaciones Científicas y Técnicas, Argentina) under Project PIP 1679.


# References
