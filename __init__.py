import pkgutil
import importlib
import logging
import sys

from . import configs
# Configure basic logging to see the errors
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + "."):
    try:
        importlib.import_module(module_name)
    except Exception as e:
        # LOGGING: Print the exact error and the module that caused it
        logging.error(f"Failed to import module '{module_name}': {e}")
        # Optionally, re-raise the error if you want the main package import to fail
        # raise

__version__ = "0.0.7"

