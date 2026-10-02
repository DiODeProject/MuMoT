"""Package version, kept in its own module so that any MuMoT module can import
it without importing the ``mumot`` package itself (avoiding a circular import)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('mumot')
except PackageNotFoundError:
    # running from a source tree that has not been installed
    __version__ = 'unknown'
