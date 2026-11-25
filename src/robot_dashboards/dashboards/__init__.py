"""Every dashboard, one module each; importing this registers them all.

WHY discovery instead of an import list: lanes add dashboards in parallel, and a shared list is a merge conflict.
"""

import importlib
import pkgutil

for _module in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
    if not _module.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module.name}")
