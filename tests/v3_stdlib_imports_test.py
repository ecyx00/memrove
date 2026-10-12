"""Packaged runtime code imports only the standard library and its own modules.

The installer promises Python 3.11+ and nothing else. An optional backend (for example a local
model server) must stay an external process: its Python packages are never imported here.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# The Python files the installer ships.
PACKAGED = sorted([ROOT / 'scripts' / name for name in ('install_v3.py', 'memrove_v3.py', 'memrove_entry.py')]
                  + list((ROOT / 'template/.claude/scripts').glob('memrove_v3*.py')))
# Shipped too, but run by its own venv (mlx), never imported by the stdlib runtime: outside the import guard.
MODEL_DAEMON = [ROOT / 'template/.claude/scripts/memrove_modeld.py']


def imported(path):
    """Every absolute import in the file, including lazy imports inside functions."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            yield node.lineno, node.module.split('.')[0]


# A local import must name a module the release actually ships. Helpers such as
# _portalock.py live next to the V3 modules in the source tree but are not packaged,
# so importing one would pass every source-tree test and fail in an installed vault (#151).
LOCAL = {path.stem for path in PACKAGED}


def allowed(name):
    return name in sys.stdlib_module_names or name in LOCAL


class StdlibImportTest(unittest.TestCase):
    def test_packaged_runtime_has_no_third_party_import(self):
        self.assertGreater(len(PACKAGED), 20)
        offenders = [f'{path.relative_to(ROOT).as_posix()}:{line} {name}'
                     for path in PACKAGED for line, name in imported(path) if not allowed(name)]
        self.assertEqual(offenders, [])

    def test_the_guard_rejects_model_runtimes(self):
        for name in ('torch', 'laya', 'transformers', 'numpy', 'requests'):
            self.assertFalse(allowed(name), name)

    def test_the_guard_rejects_unpackaged_local_helpers(self):
        for name in ('_portalock', 'flush', 'compile', 'antigravity_hooks', 'memrove_helper'):
            self.assertFalse(allowed(name), name)



if __name__ == '__main__':
    unittest.main()
