<div align="center">

# pyCC.id
Library to find interpretable models for nonlinear system identification based on the concept of characteristic curves (CCs)

https://github.com/FedejGon/pyCC.id

# pyCC :  High-Performance System Identification using Characteristic Curves

| **colab demo** | **Forums** | **Paper** | 
|:---:|:---:|:---:|
|[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/drive/136FvEwMsxLayhimgtI4Jx_IWR8l-dy-s)|[![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/FedejGon/pyCC.id/discussions)|[![Paper](https://img.shields.io/badge/arXiv-2305.01582-b31b1b)](https://arxiv.org/)||

</div>

# Instalation with pip (recommended)

## ✅ Installation

### Pip (recommended for users)
PyCC uses Symbolic Regression in some stages. We advise to install both packages in order to use all the functionalities. You can install PyCC and PySR with pip:
```bash
pip install pycc.id
pip install pysr
```

### From source (for developers)
Download or clone the repository and install with:
```bash
pip install -e .
```

## Usage
PyCC package is imported using:
```bash
import pycc
import pysr
```
(⏱️ Please note that the initial import may take a few minutes to set everything up.)


## Test
To test PyCC, there are several Tutorials available. You can download or copy them to your local machine or a cluster. Then, simply execute them, for example:
```bash
python Tutorial1.py
```

## Usage
model1, model2 = pycc.train_nn_models(t, x, x_dot, x_ddot, F_ext)
F_pred = pycc.predict(model1, model2, x, x_dot)




---






## How to cite this package

General reference to this package:
**Gonzalez2025code**



In case of using NN approach, additionally cite **Gonzalez2025nody** and **Gonzalez2025symmetries**:
  - Gonzalez, F. J. and Lara, L. P. "[Interpretable neural network system identification method for two families of second-order systems based on characteristic curves](https://doi.org/10.1007/s11071-025-11744-6)." Nonlinear Dyn. (2025)
  

In case of using Polynomial (Poly) approach, additionally cite **Gonzalez2023** and **Gonzalez2024**: 
  - Gonzalez, F.J. "[Determination of the characteristic curves of a nonlinear first order system from fourier analysis](https://doi.org/10.1038/s41598-023-29151-5)." Sci. Rep., vol. 13, 1955, (2023).
  - Gonzalez, F.J. "[System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems](https://doi.org/10.1007/s11071-024-09890-4)." Nonlinear Dyn. 112, 16167–16197 (2024). 

In case of using post-SR and/or SymbReg-CC approaches, additionally cite **Cranmer2023PySR**
  - Cranmer, M. "[Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl](https://doi.org/10.48550/arXiv.2305.01582)." arXiv preprint arXiv:2305.01582 (2023).


## Citation BibTex


```bibtex
@article{Gonzalez2023,
  title     = {Determination of the characteristic curves of a nonlinear first order system from Fourier analysis},
  author    = {Gonzalez, Federico J.},
  journal   = "Sci. Rep.",
  publisher = "Springer Science and Business Media LLC",
  volume    =  13,
  number    =  1,
  pages     = "1955",
  month     =  feb,
  year      =  2023,
  doi =   "10.1038/s41598-023-29151-5",
}

@article{Gonzalez2024,
  title = {System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems},
  author = {{F. J. Gonzalez}},
  volume = {112},
  ISSN = {1573-269X},
  url = {},
  DOI = {10.1007/s11071-024-09890-4},
  number = {18},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  year = {2024},
  month = jul,
  pages = {16167–16197}
}

@article{Gonzalez2025nody,
  title = {Interpretable neural network system identification method for two families of second-order systems based on characteristic curves},
  volume = {},
  ISSN = {},
  url = {},
  DOI = {},
  number = {},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  author = {Gonzalez,  Federico J. and  Lara, Luis P. },
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

