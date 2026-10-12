#!/usr/bin/env python3
"""Vault entry point: runs the installed CLI against the state directory the installer recorded."""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv) or ["doctor"]
    vault = Path(__file__).resolve().parent
    try:
        runtime = vault / ".memrove-runtime.json"
        if not runtime.is_file():
            raise ValueError("runtime config missing; run scripts/install_v3.py for this vault")
        state = json.loads(runtime.read_text(encoding="utf-8"))["state"]
        scripts = vault / ".claude/scripts"
        sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location("memrove_installed_cli", scripts / "memrove_v3_cli.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        return cli.main(["--vault", str(vault), "--state", str(state), *argv])
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
