#!bin/bash

conda activate docs_sphinx

make html 

# automatic update of website
sphinx-autobuild source build/html


