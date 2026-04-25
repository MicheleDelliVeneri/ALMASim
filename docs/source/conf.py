import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import almasim  # noqa: E402

project = "ALMASim"
copyright = "2024–2026, Michele Delli Veneri"
author = "Michele Delli Veneri"
release = almasim.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# Mock heavy optional deps so autodoc works on ReadTheDocs without CASA
autodoc_mock_imports = [
    "casatools",
    "casatasks",
    "casadata",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
root_doc = "index"

myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "colon_fence",
]

suppress_warnings = [
    "myst.xref_missing",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


def setup(app):
    app.add_css_file("custom.css")


intersphinx_mapping = dict(
    h5py=("https://docs.h5py.org/en/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    astropy=("https://docs.astropy.org/en/stable/", None),
)
