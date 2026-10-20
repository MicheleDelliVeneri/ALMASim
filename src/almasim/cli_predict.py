"""Alias of :mod:`almasim.cli.predict` (MADIV-28 T3.1); removed in T8.4.

The module itself is installed under this name, so ``from almasim import
cli_predict`` and patches of ``almasim.cli_predict.<name>`` reach the real code.
"""

import sys

from almasim.cli import predict as _module

sys.modules[__name__] = _module
