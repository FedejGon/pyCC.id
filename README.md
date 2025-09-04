# pyCC
Library to find interpretable models for nonlinear system identification based on the concept of characteristic curves (CCs)



## Usage

```python
import pycc

model1, model2 = pycc.train_nn_models(t, x, x_dot, x_ddot, F_ext)
F_pred = pycc.predict(model1, model2, x, x_dot)




---

## ✅ How to Install Locally

In your terminal, from the outer `library_python/` folder:

```bash
pip install -e .




