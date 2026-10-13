# memrove-chat

Claude Code function mod (early access, CC 2.1.288+). It is the live driver of memrove's chat memory: the transport only. All memory logic stays in the Python CLI.

- After every main-loop turn that ended with an answer it runs, in the background and one run at a time: `memrove chatlog ingest`, then up to `batches` (3) times `chatlog prepare --json --no-ingest` -> `$.model.complete({model:'haiku', system, prompt})` -> `chatlog apply <batch_id>` (raw reply on stdin). A failed model call is handed back with `apply --error`. After 3 failed runs it pauses for 10 minutes. The status line shows `chat memory: building batch 1 · 3508 nodes · 1/60 calls today`, or the error.
- Registers the model tools `zoom(id, n[, page])` and `date(id)` (`mcp__memrove-chat__zoom`, `...__date`), each calling `memrove zoom` / `memrove date`. `project([name])` calls `memrove project`: latest decisions (superseded marked), open loops and the last session's end state, pull only.
- Adds the whole view (`memrove wake --all`, up to 32 KB by default) as a `prompt.context` block named `memroveChat` (startup, resume, /clear, after compaction). The memrove SessionStart instruction shrinks to one line when that worked.
- `fresh` (default off, experimental): after a turn `$.session.compact()` runs and the `session.compact` handler returns the last user message and everything after it, verbatim; the view comes back as the `prompt.context` block. Interactive sessions only (`$.session.compact()` is unavailable in `claude -p`). Measured: each fresh turn writes ~44k cache tokens, against ~0.3k with fresh off, so leave it off.

## Options (userConfig)

| option | default | meaning |
|---|---|---|
| `memrove` | `memrove` | entry point, split on spaces, no quoting. Example: `python3 /path/scripts/memrove_v3.py --vault V --state S` |
| `batches` | 3 | batches (Haiku calls) per turn |
| `model` | `haiku` | compactor model alias |
| `fresh` | false | fresh context per turn |

## Develop

```sh
npm install
npx tsc -p . && npx eslint --max-warnings 0 .
claude plugin test .
claude --plugin-dir . 
```

Install from the directory marketplace in this repo (`mods/.claude-plugin/marketplace.json`, checked with `claude plugin validate mods`):

```sh
claude plugin marketplace add /path/to/memrove/mods
claude plugin install memrove-chat@memrove-mods
```

Types come from `/plugin-types` (`.claude-plugin/types/`, git-ignored).
