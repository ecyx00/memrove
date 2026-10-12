"""Environment helpers for tests that run the hook or the installer in a child process."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

# Every MEMROVE_* variable is a user setting or a hook flag. A suite run from a configured shell must not
# inherit them: only values the suite itself inherited are dropped, so one a test sets still reaches its child.
_INHERITED = {k: v for k, v in os.environ.items() if k.upper().startswith("MEMROVE_")}


def inherited_env(**extra):
    """The current environment without the inherited MEMROVE_* settings, plus extra."""
    env = {k: v for k, v in os.environ.items() if _INHERITED.get(k) != v}
    env.update(extra)
    return env


def clean_environ(**extra):
    """In-process counterpart of inherited_env: a patch.dict for os.environ."""
    return patch.dict(os.environ, inherited_env(**extra), clear=True)


def isolated_env(home):
    """A child-process environment confined to a throwaway home directory."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    env = {"HOME": str(home), "USERPROFILE": str(home), "APPDATA": str(home / "appdata"),
           "LOCALAPPDATA": str(home / "localappdata"), "TEMP": str(home), "TMP": str(home),
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "MEMROVE_V3_NO_SPAWN": "1"}
    env.update({k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ})  # Windows needs both
    return env
