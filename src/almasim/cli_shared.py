"""Alias of :mod:`almasim.cli._shared` (MADIV-28 T3.1); removed in T8.4.

The module itself is installed under this name, so ``from almasim import
cli_shared`` and patches of ``almasim.cli_shared.<name>`` reach the real code.
"""

import sys

from almasim.cli import _shared as _module

sys.modules[__name__] = _module
