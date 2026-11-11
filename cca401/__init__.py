"""Maintained native CCA-Zoo 4.0.1 computation for this project."""
from importlib.metadata import version

if version("cca-zoo") != "4.0.1":
    raise RuntimeError("Activate py314 with cca-zoo==4.0.1 for current analysis")
