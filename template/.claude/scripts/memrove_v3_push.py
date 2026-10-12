"""Machine-local switch for context pushed into Claude sessions; defaults off.

`recall_cards`: per-prompt recall cards.
Stored in `<state>/push.json`, never in `.memrove-preferences.json` (an older release rejects unknown
preference keys, so a rollback would break). Missing or invalid file means off. The chat
memory instruction, release notice, receipt reminders and capture do not depend on these.
"""
import json
import os
from pathlib import Path
import tempfile

FILE = 'push.json'
KEYS = ('recall_cards',)


def read(state):
    try:
        data = json.loads((Path(state) / FILE).read_text(encoding='utf-8'))
        return {key: data.get(key) is True for key in KEYS}
    except (OSError, ValueError, AttributeError):
        return {key: False for key in KEYS}


def save(state, changes):
    value = dict(read(state), **changes)
    Path(state).mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.push-', dir=state)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(value, out)
        os.replace(temporary, Path(state) / FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value
