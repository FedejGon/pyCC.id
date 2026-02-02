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

   pip install numpy scipy matplotlib torch
   pip install pysr
   pip install pycc.id

In Google Colab notebook and Jupiter notebook, add ! at the beginning of the lines as follows 

.. code-block:: bash
   !pip install numpy scipy matplotlib torch
   !pip install pysr
   !pip install pycc.id

.. note::

   This code and its documentation in this stage are continuously being improved with new features.
   If you encounter any problems, it could be because you're using an older version of pyCC.id.
   In that case, we advise to upgrade the code with the command: (sometimes you need to run it two times)

   .. code-block:: bash

      pip install --upgrade pycc.id
      #for colab and jupiter, add ! at the beginning

Installation for developers (from source)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download or clone the repository and install with:

.. code-block:: bash

   pip install -e .

 
.. note::
   **Initial import delay**: The first time you run ``import pycc``, it may take ∼3 minutes to set up dependencies. Subsequent imports will be fast.

-------------------------------------------


