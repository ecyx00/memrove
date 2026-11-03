"""Reader: Antigravity brain/<sid>/.system_generated/logs/overview.txt (same step_index JSONL as transcript*.jsonl, which the live log skips)."""
from pathlib import Path

import memrove_v3_chatlog as chatlog

NAME = "antigravity-overview"


def files(home):
    return sorted(Path(home).glob(".gemini/antigravity/brain/*/.system_generated/logs/overview.txt"))


def read(path):
    """-> msgs in chatlog-source form ({ts, kind, text, k, final})."""
    return chatlog.antigravity_source(path, 0)[0]
