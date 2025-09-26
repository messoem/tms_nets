import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'tms_nets'
copyright = '2025, Arsenii Sychev'
author = 'Arsenii Sychev'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Автоматически импортирует документацию из docstrings
    'sphinx.ext.viewcode', # Добавляет ссылки на исходный код
    'sphinx.ext.napoleon', # Понимает docstrings в стиле Google и NumPy
]
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
