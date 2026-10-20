"""The installed almasim version, read from the package metadata.

pyproject.toml is the only place the version is written (the release workflow
checks the tag against it); a hard-coded copy here had drifted to 2.1.10 while
the package was 2.1.16.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("almasim")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"
