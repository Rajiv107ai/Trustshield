# Gravitas Reliability Protocol Enforcement

Always apply the **Gravitas** engineering reliability runtime (`.agents/skills/gravitas/SKILL.md`) for all implementation, debugging, refactoring, review, and verification tasks in this workspace:

1. **Ground before mutating:** Inspect target files, tests, and callers before proposing or writing code changes. Never act on assumed file state.
2. **Stay in declared scope:** Limit modifications strictly to files within the declared scope of the task.
3. **Completion requires post-mutation evidence:** An issue or task is only complete when verified by real command output (tests, lint, type checkers) executed *after* the latest file edit. Stale output or self-review does not count as completion proof.
4. **Never exact-retry a failure:** When an edit or test fails, diagnose the root cause and adapt strategy before retrying.
5. **Satisfy acceptance criteria:** Verify all explicit requirements and contracts before concluding.
6. **Acknowledge uncertainties & limitations:** Transparently report any unverified assumptions, warnings, or environment constraints.
7. **Consistent reporting structure:** Conclude every non-trivial task with:
   - **What Changed:** Files modified and behavioral delta.
   - **Evidence:** Actual observed terminal outputs and test counts.
   - **Why:** Technical rationale for the decisions made.
