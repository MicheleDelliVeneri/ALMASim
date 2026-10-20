"""Alias of :mod:`almasim.cli.products` (MADIV-28 T3.1); removed in T8.4.

The module itself is installed under this name, so ``from almasim import
cli_products`` and patches of ``almasim.cli_products.<name>`` reach the real code.
"""

import sys

from almasim.cli import products as _module

sys.modules[__name__] = _module
