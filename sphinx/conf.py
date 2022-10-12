# Configuration file for the Sphinx documentation builder

import inspect
import importlib

## Project

package = 'zuko'
project = 'Zuko'
copyright = '2022, François Rozet'
repository = 'https://github.com/francois-rozet/zuko'

## Extensions

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
}
autodoc_inherit_docstrings = False
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'

autosummary_ignore_module_all =  False

intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}


def linkcode_resolve(domain: str, info: dict) -> str:
    module = info.get('module', '')
    fullname = info.get('fullname', '')

    if not module or not fullname:
        return None

    objct = importlib.import_module(module)
    for name in fullname.split('.'):
        objct = getattr(objct, name)

    try:
        file = inspect.getsourcefile(objct)
        file = file[file.rindex(package) :]

        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1
    except Exception as e:
        return None
    else:
        return f'{repository}/tree/docs/{file}#L{start}-L{end}'


napoleon_custom_sections = [
    ('Shapes', 'params_style'),
    'Wikipedia',
]

## Settings

add_function_parentheses = False
default_role = 'literal'
exclude_patterns = ['templates']
html_copy_source = False
html_css_files = [
    'custom.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
]
html_favicon = 'static/logo.svg'
html_logo = 'static/logo.svg'
html_show_sourcelink = False
html_sourcelink_suffix = ''
html_static_path = ['static']
html_theme = 'furo'
html_theme_options = {
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': repository,
            'html': '<i class="fa-brands fa-github fa-lg"></i>',
            'class': '',
        },
    ],
    'light_css_variables': {
        'color-api-keyword': '#007020',
        'color-api-name': '#0e84b5',
        'color-api-pre-name': '#0e84b5',
    },
    'dark_css_variables': {
        'color-api-keyword': '#66d9ef',
        'color-api-name': '#a6e22e',
        'color-api-pre-name': '#a6e22e',
    },
    'sidebar_hide_name': True,
}
html_title = project
pygments_style = 'sphinx'
pygments_dark_style = 'monokai'
rst_prolog = """
.. role:: py(code)
    :class: highlight
    :language: python
"""
templates_path = ['templates']
