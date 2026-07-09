# Server Workspace

This directory is intentionally empty in git.

Use it for server-side temporary outputs that should not be committed, for example:

- AutoBots smoke-test HDF5 files
- converted generated datasets
- short debug runs

Scripts use this directory instead of `/tmp` when a persistent local working path is useful.
