# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyCC.id'
copyright = '2025, Federico J. Gonzalez'
author = 'Federico J. Gonzalez'
release = '0.1.0'


import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # add repo root so autodoc can import your package


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = []
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',   # create summary pages / stubs
    'sphinx.ext.napoleon',      # Google/Numpy style docstrings (optional)
    'sphinx.ext.viewcode',      # link to source
    'sphinx_autodoc_typehints', # optional: show type hints
]

autosummary_generate = True   # generate autosummary .rst files automatically
autodoc_member_order = 'bysource'


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
