==============
Practical Tips
==============

----------------------------
NN-CC: The Core Method
----------------------------


The primary and most flexible identification method in ``pyCC`` is **NN-CC** (Neural Network - Characteristic Curves). This method uses neural networks to approximate the unknown functions :math:`\{\mathbf{f}\}` and can simultaneously identify unknown scalar parameters :math:`\mathbf{a}` as part of a hybrid model.

A key strategy for improving the accuracy and physical consistency of the identified model is to **incorporate prior physical knowledge**. This is a core strength of ``pyCC``.

You can inject this information using the ``constraints`` parameter during training. For example, if you know a restoring force :math:`f_2(x_1)` must be an odd function (e.g., :math:`f_2(-x_1) = -f_2(x_1)`) and must pass through the point (1,2), you can specify this:

.. code-block:: python

   constraints = [
       {'constraint': 'f2 odd'},
       {'constraint': 'f2(1)=2'},
   ]

Adding constraints like **symmetries** (odd/even) or **known values** (e.g., :math:`f(0)=0`) significantly aids the discovery process, reduces the space of possible solutions, and leads to more robust and physically-grounded results.

Additionally, a powerful workflow is to apply **Symbolic Regression as a post-processing step** to the characteristic curves :math:`\{\mathbf{f}\}` obtained from a trained NN-CC model. This has been found to be highly useful not only for discovering a final, simple **analytical expression** for the functions but also for **reducing potential overfitting** that might be present in the raw neural network fitting.

-----------------------------------------------
Alternative Methods: Poly-CC and SymbR-CC
-----------------------------------------------

While NN-CC is the core method, ``pyCC`` also provides other approaches for comparison and specific use cases:

* **Poly-CC (Polynomials)**: This method uses polynomial basis functions (e.g., :math:`f(x) = c_1 x + c_2 x^2 + ...`) to model the characteristic curves. It is less flexible than NN-CC but can be useful for simple systems or for establishing a baseline identification.

* **SymbR-CC (Symbolic Regression)**: This method, which leverages the ``pySR`` library, attempts to find an explicit symbolic mathematical expression for the characteristic curves (e.g., :math:`f(x) = 0.5 \tanh(500x)`).

.. warning::
   Be cautious when using SymbR-CC with complex model structures (:math:`\mathbf{G}`). Symbolic regression can struggle or fail if the proposed equation involves complex operations. For example, it may have issues if an unknown function :math:`f(x)` is located in the **denominator** of an expression or inside another complex function. It performs best when the unknown functions are combined in simpler, additive, or multiplicative ways.
   
.. note::
   Often, performing simulations with SymbR-CC is slow, as it requires to evaluate the analytical expressions for the functions at every time step. Aternatively, you can simulate the system using the ``Interp`` method and using as input the ``evals`` variable obtained from SymbR-CC. The simulation using interpolation is much faster than analytical evaluation.  
