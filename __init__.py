import pkgutil
import importlib
import logging
import sys

from . import configs
# Configure basic logging to see the errors
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + "."):
    importlib.import_module(module_name)


__version__ = "0.0.7"

