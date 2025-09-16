# pyCC.id
Library to find interpretable models for nonlinear system identification based on the concept of characteristic curves (CCs)


```instalation
pip install pycc.id

```instalation in Google Colab or Jupiter
!pip install pycc.id


## Test
python program.py


## Usage
model1, model2 = pycc.train_nn_models(t, x, x_dot, x_ddot, F_ext)
F_pred = pycc.predict(model1, model2, x, x_dot)




---

## ✅ How to Install Locally from source 

Download source, from the root directory run:

```bash
pip install -e .




