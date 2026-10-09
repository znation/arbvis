# Principles

Design principles this project holds — the codified answer to "what would a senior engineer on
this team always do." Every loop's prompt carries these; uphold them in everything you produce.
Only the director and steward roles may edit this file. Phrase new principles positively: state
what to do, not what to avoid. This list is a starting point: the director and steward own it
and will tune it to this project.

- Prefer the standard library over a new dependency.
- Keep each file focused on one responsibility, and small enough to read in one sitting.
- Every behavior change ships with a test.
- Small, complete, and correct beats big and half-done: one focused change per tick. Focused
  means one theme, not one site: a kind of fix applied everywhere it holds is one change, and
  the same fix landed one site per tick spends a review and a landing on each.
