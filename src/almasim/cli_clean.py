"""Alias of :mod:`almasim.cli.clean` (MADIV-28 T3.1); removed in T8.4.

The module itself is installed under this name, so ``from almasim import
cli_clean`` and patches of ``almasim.cli_clean.<name>`` reach the real code.
"""

import sys

from almasim.cli import clean as _module

sys.modules[__name__] = _module
