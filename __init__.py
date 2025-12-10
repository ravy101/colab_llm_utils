import pkgutil
import importlib
import logging
import sys
import os

# 1. Keep your manual imports
from . import configs

# Configure logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

# 2. Get the current package object so we can attach things to it
current_package = sys.modules[__name__]

# 3. The Corrected Loop
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Skip the current package itself if it appears to avoid recursion
    if module_name == __name__:
        continue

    # A. Import the module (Load into memory)
    # Note: We use relative import syntax (e.g., .metric) to ensure it stays inside this package
    full_name = f"{__name__}.{module_name}"
    imported_module = importlib.import_module(full_name)
    
    # B. THE MISSING LINK: Attach it to the package!
    # This makes 'colab_llm_utils.metric' exist.
    setattr(current_package, module_name, imported_module)

# 4. Your __all__ definition is fine, but it only affects 'from X import *'
__all__ = ["configs",
           "cascades",
           "confidence",
           "likelihood",
           "misc",
           "model_tools",
           "modelling",
           "plotting",
           "scorers",
           "text",
           "metric",
           "embedders"]

__version__ = "0.0.7"