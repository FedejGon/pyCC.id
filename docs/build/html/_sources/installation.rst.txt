============
Installation
============ 

---------------------------------
Installation with pip (Recommended)
---------------------------------

Installation for users
~~~~~~~~~~~~~~~~~~~~~~
Some features in pyCC include using the Symbolic Regression (pySR) package.
Thus we recommend installing this package first. To install both packages use:

.. code-block:: bash

   pip install pycc.id


In Google Colab or Jupyter notebook, add ! at the beginning of the line as follows:

.. code-block:: bash

   !pip install pycc.id

.. hint::

   This code and its documentation in this stage are continuously being improved with new features.
   If you encounter any problems, it could be because you're using an older version of pyCC.
   In that case, we advise to upgrade the code with the command: (sometimes you need to run it two times)
   
   .. code-block:: bash

      pip install --upgrade pycc.id
      #for colab and jupyter, add ! at the beginning

Once installed, the pycc library can be imported with: 

.. code-block:: bash  

   import pycc

See the `Usage` and `Examples` tabs for examples and more information.

.. note::
   **First-time import**: The first time you run ``import pycc``, it may take ∼3 minutes to set up dependencies (including ``torch``, ``numpy``, ``scipy``, ``matplotlib``, and ``pysr``). Subsequent imports will be nearly instantaneous.

Installation from source (for developers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download or clone the repository and install locally with:

.. code-block:: bash

   pip install -e .

 
-------------------------------------------


