# Deviations and notes: protocol 2

Logged under prereg-2/PREREGISTRATION.md §11. The freeze commit is `c2474be59c90df62c40762a3a6d438269d3a1657`, and its
files are never edited.

## E1: stale status label (editorial; no change to content), 27 Sep 2026

At the freeze commit, the header of `PREREGISTRATION.md` still reads "**Status:** DRAFT, not frozen", and the first
comment line of `config/prereg.yaml` still reads "DRAFT until the freeze commit". The label was not updated before
committing.

The same header sentence defines the freeze point, "frozen at the first commit that contains this file", which is
`c2474be`. So which text binds is unambiguous. **Both files are frozen as committed.** Every threshold, task, seed
and rule in them is binding. The label is left as it is, because editing a freeze commit, even cosmetically, is what
the freeze exists to prevent.

When this note was written, no model had been run and no §5 statistic had been computed on any model.
