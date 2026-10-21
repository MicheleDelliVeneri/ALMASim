# Proposal: one console for everyone, through slurmrestd

*For the cluster's administrators. Nothing in ALMASim depends on it: the
console works today without it.*

## Today

Each user starts their own console on the login node (`almasim serve`) and
reaches it through an SSH tunnel. The console runs as that user: it submits
jobs with `sbatch`, browses only what that user may read, and keeps nothing
but the run folders (shared by the team if the site profile says so). It
needs nothing from the administrators beyond a login node and Slurm's
command-line tools.

What that costs users: starting a console and a tunnel each session, one
console per person on the login node (small: an idle Python process), and
no way to reach it without SSH.

## The option

One console service for everyone, reached in a browser through the
institute's single sign-on, which submits each user's jobs **as that user**
through Slurm's REST daemon, `slurmrestd`, authenticated with JSON Web
Tokens (`auth/jwt`).

```
browser ──SSO──▶ console hub ──HTTPS + user JWT──▶ slurmrestd ──▶ slurmctld
                    │                                              │
                    └── run folders (group-shared) ◀── jobs run as the user
```

## What it needs from the cluster

zwicky runs Slurm 22.05.9 with munge authentication only; neither
`slurmrestd` nor JWT is enabled. The hub needs:

1. **JWT authentication** alongside munge: `AuthAltTypes=auth/jwt` and
   `AuthAltParameters=jwt_key=/etc/slurm/jwt_hs256.key` in `slurm.conf`
   (and `slurmdbd.conf`), the key readable by `SlurmUser` only, and a
   restart of `slurmctld` and `slurmdbd`.
2. **`slurmrestd`** running as a dedicated unprivileged user (not root, not
   `SlurmUser`), on a socket or port only the hub can reach. 22.05 serves
   the `v0.0.38` REST API: submit, job state, cancel; enough for the hub.
3. **Per-user tokens.** Two ways, with different trust:
   - each user makes their own (`scontrol token lifespan=…`) and gives it to
     the hub once per lifespan. No new trust, but a chore and a secret to
     paste;
   - or the hub, as a trusted service, mints tokens for the user who signed
     in (`scontrol token username=<user>`, which needs `SlurmUser` or root).
     Smooth, but the hub can then act as any user: it becomes part of the
     cluster's security boundary and needs review, hardening and logging
     like one.
4. **A host** for the hub (a service VM or node) that reaches `slurmrestd`
   and mounts the shared filesystem, and an SSO client registration.

## What changes in ALMASim

- A Slurm executor that speaks REST (submit, poll, cancel) beside today's
  `sbatch` one; the run folders, manifests and the wrapper stay as they are.
- Sign-in through SSO instead of the console's token, and the user's
  identity on every request.
- Files: the hub runs as one service account, so it can only show and write
  what that account may: in practice the group-shared areas (the runs root,
  the project folders). Browsing a user's home would need the hub to act as
  the user on the filesystem too (impersonation), which this proposal does
  not suggest.
- Previews and other work the console does itself must move into jobs
  entirely (they mostly have: U3.7), since one process would serve everyone.

## An alternative: Open OnDemand

[Open OnDemand](https://openondemand.org/) gives each user, after SSO, their
own web apps started as Slurm jobs and proxied to the browser, without JWT
or `slurmrestd`. Today's console fits it almost as it is: an "interactive
app" that runs `almasim serve --port …` in a job (or on the login node) and
lets OnDemand's proxy carry the token. Users get a browser link instead of
a tunnel, every action stays theirs, and the cluster gains a service many
centres already run for Jupyter and RStudio.

## Recommendation

Keep the per-user console as the supported way (it needs nothing). If a
browser-only entry is wanted, **Open OnDemand** is the smaller step and keeps
every action under the user's own account. A `slurmrestd` hub is worth it
only if a single shared service is itself the goal; it needs the
administrators' agreement on JWT, on who mints tokens, and on running a
service that submits on users' behalf.
