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
date: 25 November 2025
bibliography: paper.bib
---


# Summary

`pyCC.id` is a Python package for data-driven system identification that provides a flexible "grey-box" framework for discovering interpretable, nonlinear governing equations from time-series data. It is designed to bridge the gap between opaque black-box models (e.g., Neural ODEs) and restrictive, library-dependent methods (e.g., SINDy [@Brunton2016]).

The core philosophy of `pyCC.id` is to decompose a complex dynamical system, $d\mathbf{x}/dt = \mathbf{F}(\mathbf{x}, t)$, into a user-defined model structure, $\mathbf{G}$. This structure explicitly separates measured inputs (like system state $\mathbf{x}$ and external forces) from the unknown model components.  These components are: (1) a set of unknown one-dimensional functions ($\{\mathbf{f}\}$) referred to as **Characteristic Curves** (CCs), which capture the underlying system nonlinearities (such as stiffness or damping), and (2) a set of unknown scalar parameters ($\mathbf{a}$).

The package offers a flexible, multi-backend approach to identify these unknown functions and scalar parameters. Users can model the characteristic curves $\{\mathbf{f}\}$ using powerful non-biased approximators like neural networks (NN-CC, via `PyTorch` [@Paszke2019pytorch], and compatible with both CPU and GPU devices), simple polynomial basis functions (Poly-CC), or discover analytical expressions directly using symbolic regression (SymbR-CC, via `PySR` [@Cranmer2023PySR]). A distinctive feature of this framework is that physical prior knowledge (e.g., symmetries, conservation laws) can be easily and directly incorporated as constraints, guiding the discovery process toward physically consistent models for experts in engineering, physics, and biology.


# Statement of Need

Researchers in science and engineering often face a trade-off in system identification. **Black-box models**, such as standard Neural ODEs[@Chen2019], can achieve high predictive accuracy but are opaque, offering little physical insight[@Wu2025]. Conversely, **interpretable "white-box" methods**, like SINDy [@Brunton2016] or pure Symbolic Regression [@Cranmer2023PySR], aim to find simple analytical equations. However, their success often depends on the true dynamics being sparsely represented in a pre-defined library of candidate functions or, in the case of SR, can be computationally challenging and highly sensitive to hyperparameter tuning when applied to full, complex systems.

`pyCC.id` is built to fill this "grey-box" gap, targeting domain experts who possess partial physical knowledge of their system (i.e., the model structure $\mathbf{G}$) but need to discover the specific functional forms of its components (the characteristic curves $\{\mathbf{f}\}$). It addresses the need for a tool that can take advantage of  powerful, non-biased and non parametric function approximators (e.g., neural networks) within a physically constrained, interpretable framework, rather than forcing an all-or-nothing choice between accuracy and interpretability.



# Comparison to other packages

`pyCC.id` integrates concepts from several existing tools but applies them within a unique, structured framework:

* **SINDy (e.g., `pysindy`)**: SINDy [@Brunton2016] excels when the true dynamics are a sparse combination of terms from a user-provided candidate library. `pyCC.id` differs by not relying on a pre-defined library for the full dynamics. Instead, its NN-CC backend learns the shape of unknown 1D functions, which can then be analyzed, offering flexibility when the underlying functions (e.g., complex friction or stiffness) are not easily represented by simple library terms.

* **Symbolic Regression (e.g., `PySR`)**: While `pyCC.id` uses `PySR` [@Cranmer2023PySR] as a backend (SymbR-CC) and a post-processing tool, its application is distinct. Rather than applying SR directly to the full, high-dimensional derivative data (a computationally difficult task), `pyCC.id`'s primary workflow first isolates the unknown components as simple 1D functions (using NN-CC) and *then* applies SR to these much simpler, cleaner 1D curves to find their analytical forms.

* **Black-Box Neural ODEs**: Standard Neural ODE packages learn the entire derivative function $\mathbf{F}$ as a single, monolithic neural network. `pyCC.id` employs a "grey-box" approach, using `PyTorch` [@Paszke2019pytorch] to model only the specific, interpretable 1D components $\{\mathbf{f}\}$ within a user-defined physical structure $\mathbf{G}$, making the resulting model inherently interpretable.

* **Physics-Informed Neural Networks** (PINNs)[@Raissi2019] represent a powerful approach to simulate systems (e.g., PDEs), and the incorporation of constraints into the loss function during the neural network training. While excellent for solving or parameterizing known or partially-known governing equations, they are typically less suited for the explicit discovery of the functional form of complex, coupled system components, which often remains a black-box function within the network itself.

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

* **Interpretable Models**: Decomposes complex, high-dimensional dynamics into a set of simple, 1D characteristic curves that often have direct physical meaning (e.g., stiffness, damping), and a set of scalar parameters (e.g., mass value).
* **Flexible Function Parametrization**: Supports multiple backends for modeling the characteristic curves, allowing users to choose the right tool for their problem:
    * **Neural Networks (NN-CC)**: Uses `PyTorch` [@Paszke2019pytorch] for high-performance, non-biased function approximation.
    * **Polynomials (Poly-CC)**: Provides a simple baseline using polynomial basis functions.
    * **Symbolic Regression (SymbR-CC)**: Uses `PySR` [@Cranmer2023PySR] to discover analytical expressions directly.
* **Physics-Informed Discovery**: Allows users to inject domain knowledge as constraints during training (e.g., `'f1 odd'`, `'f2(0)=0'`) or by defining conserved quantities in the loss function. This leads to more robust and physically consistent models.
* **Hardware Acceleration**: Natively supports multicore CPUs and GPUs from both NVIDIA (`cuda`) and Intel (`xpu` via `intel-extension-for-pytorch`) for accelerating neural network training.
* **Built-in Simulator**: Includes a versatile ODE simulator (`pycc.simulate`) compatible with all identified model types for validation and analysis.
* **Comprehensive Documentation**: Provides a Google Colab notebook for a quick start, as well as a full gallery of tutorials and examples in the repository.





# Mentions of scholarly publications

The `pyCC.id` package provides a generalized software implementation of the CC-based approaches for system identification. This core methodology was central to methods first introduced for first-order systems [@Gonzalez2023; @Gonzalez2024] and later extended to second-order systems [@Gonzalez2025nody; @Gonzalez2025arxiv]. pyCC.id unifies and extends this approach into a single framework capable of handling higher-order dynamical systems.



# Key References

The software package is available at: [https://github.com/FedejGon/pyCC.id](https://github.com/FedejGon/pyCC.id)

# Acknowledgements

This work was partially supported by CONICET (Consejo Nacional de Investigaciones Científicas y Técnicas, Argentina).
We acknowledge the computational resources from the Clementina XXI supercomputer and CCT-Rosario Computational Center, both managed by the High Performance Computing National System (SNCAD, ME-Argentina), with the support of the Undersecretariat of Science and Technology of Argentina.

# References
