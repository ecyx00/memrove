"""Reader: Codex rollouts. The live chat log already holds them, so the importer's skip rule makes this add 0; it proves that rule."""
from pathlib import Path

import memrove_v3_chatlog as chatlog

NAME = "codex"


def files(home):
    return sorted(Path(home).glob(".codex/sessions/**/rollout-*.jsonl"))


def read(path):
    return chatlog.codex_source(path, 0)[0]
