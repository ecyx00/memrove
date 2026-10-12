---
name: memrove-doctor
description: Check that memrove really works: hook and worker state, the pending queue, source freshness and size reports. Use for health checks and when memory seems broken.
---

# memrove doctor

Run `python3 memrove.py doctor` from the vault root. Do not copy hook lists from another setup or another user's paths.

Read the pending and failed events, the last sync, source conflicts and `status`. An exit code of 0 with an empty lifecycle history does not prove that a client is connected: say so instead of reporting it as verified. Explain the fault and the smallest fix without pasting error logs or private notes into the chat.

- Source changes that are not indexed: run `python3 memrove.py sync`.
- To prove a client is wired, write a test note that holds no private data, start a new real session and ask for the fact in that note from hook context only. That is stronger evidence than checking that the file exists. For Codex check the project and the `/hooks` trust; for Claude Code start a new session in the project. Never write trust hashes, and do not count a result obtained with a bypass as proof of a normal install.
- `companion_hygiene` shows the size of the companion files. `Last-Session.md` or `Threads.md` in `over_limit` is a hygiene finding, not a sync fault: show the plan with `python3 memrove.py companion-compact --dry-run`, then run `companion-compact` if it fits. It moves old sections word for word into `Archive/` and deletes nothing. `Rules.md`, `Core.md` and `Journal.md` grow on purpose and are only reported.
- `instruction_references` lists `[[wikilinks]]` and `[text](path)` links in AGENTS.md, CLAUDE.md, `Rules.md` and the skill folders that point at nothing in the vault, with file and line; code blocks are skipped. It is information and never sets `needs_attention`. Tell the user which instruction points at a missing note and suggest fixing the link if the target moved; do not invent a note.
- Never delete the runtime database or user files to turn an indicator green.

Tell the user three things: what works, what is unverified or broken, and the one next fix. Give a short concrete result, not raw JSON. If the user's earlier approval already covers a safe fix, do not ask again.
