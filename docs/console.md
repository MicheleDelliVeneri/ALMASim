# Web console

`almasim serve` runs the web console for one user on a cluster's login
node: it serves the pages and API v2 on a private socket (or a port),
submits every command as a Slurm job (or, where the site profile says so,
runs it on its own machine), and follows the runs live. Each user starts
their own console; the runs live in run folders that a whole team can share.

## A shared runs folder

A team sees each other's runs when the site profile names one runs root
for everyone:

```toml
# the site profile (ALMASIM_SITE_PROFILE), e.g. examples/site-profiles/zwicky.toml
runs_root = "/SKAO_data/cycle11/almasim-runs"
```

Every run is a folder `<runs root>/<day>/<run id>/` holding its manifest
(`job.json`), its status, its batch script and its logs. Each user's
console and `almasim jobs list --all` read every folder there:

- everyone **sees** every run: the Jobs page's *Team* view, and the team
  runs on the home page;
- only a run's **owner** (the user who submitted it) may cancel or delete
  it; the others get a refusal naming the owner (`almasim jobs cancel|delete`
  say the same). Anyone may start a copy of it as their own run (*Edit and
  run* on its page);
- deleting removes the run folder only (manifest, logs, and whatever the
  command wrote into it); outputs written elsewhere stay.

### Setting the folder up

The runs root has to be writable by the whole group, and stay so for the
day folders and run folders that users' consoles create in it:

```bash
# once, by whoever administers the shared area (here group "skao")
mkdir -p /SKAO_data/cycle11/almasim-runs
chgrp skao /SKAO_data/cycle11/almasim-runs
chmod 2775 /SKAO_data/cycle11/almasim-runs   # setgid: new folders inherit the group
```

- The **setgid bit** (`2` in `2775`) makes every folder created inside
  belong to the group, whoever creates it.
- Jobs run with **umask 002** (the job wrapper sets it), so what a run writes
  is group-writable too; nothing needs setting in users' shells for that.
- `almasim doctor` checks that the runs root is writable and group-shared
  ("writable, group-shared (setgid)").

A user who wants their runs private sets their own runs root in their user
profile (`~/.config/almasim/almasim.toml`, or the console's Settings page):

```toml
runs_root = "~/almasim-runs"
```
