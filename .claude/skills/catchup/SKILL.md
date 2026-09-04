---
name: catchup
description: Get up to speed on the sar-oilspill codebase cheaply, or refresh CLAUDE.md after building something. Use when starting work in a new session and needing current project state, when the user asks "where are we", "what's built", "what's next", or after finishing a pipeline stage so the next session inherits the change. Reads CLAUDE.md plus a few cheap commands instead of auditing the source tree.
---

# Catch up on sar-oilspill

Two modes. Pick by what the user asked for.

## Mode A — orient (default)

`CLAUDE.md` is already in context; it loads automatically. Do **not** re-read it
and do **not** open source files to "verify" what it says.

Run only these, and only if the answer actually matters for the task at hand:

```bash
git -C sar-oilspill log --oneline -8
git -C sar-oilspill status --short
```

That is enough to know what changed since `CLAUDE.md` was last written. Open
source files only for the specific module the user's task touches.

Signs `CLAUDE.md` is stale — commits mentioning work its status table calls
"not built", or a passing-test count that no longer matches. If stale, say so
in one line and offer Mode B; do not silently work from the stale version.

## Mode B — refresh after building something

Run when a pipeline stage was just finished, or the user asks to update the
memory file.

1. Gather current state:

```bash
git -C sar-oilspill log --oneline -8
sar-oilspill/.venv/Scripts/python.exe -m pytest sar-oilspill -q 2>&1 | tail -3
```

2. Edit `sar-oilspill/CLAUDE.md` in place. Change only what actually moved:

   - **Status table** — flip the finished row to `done`, name the module paths,
     set the next row to `next`.
   - **Test count** in the line under the table.
   - **Environment facts** — only if something about the machine, the data or
     the installed packages genuinely changed.
   - **Architecture decisions** — append a bullet only for a decision a future
     session would otherwise reopen and argue about. A decision that is obvious
     from reading the code does not belong here.
   - **Open questions** — remove answered ones, add new blockers.

3. Keep it short. This file is paid for on every single session, so length is a
   real cost. Delete anything the code now states plainly. If a section has
   grown past roughly its current size, cut rather than append.

Do not restate file listings, function signatures or line counts — those go
stale immediately and the code is the source of truth for them. Record decisions
and constraints, which the code does not explain on its own.
