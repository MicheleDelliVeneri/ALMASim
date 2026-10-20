"""Alias of :mod:`almasim.cli.simulation` (MADIV-28 T3.1); removed in T8.4.

The module itself is installed under this name, so ``from almasim import
cli_simulation`` and patches of ``almasim.cli_simulation.<name>`` reach the real code.
"""

import sys

from almasim.cli import simulation as _module

sys.modules[__name__] = _module
