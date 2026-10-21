# ALMASim for Open OnDemand

An Open OnDemand interactive app that runs the ALMASim web console as a
small Slurm job of the user's, on a compute node, reached through the
portal's proxy at `/node/<host>/<port>/` (Open OnDemand plan, O4).

| File | What it does |
| --- | --- |
| `manifest.yml` | The app's name and place in the portal's menu. |
| `form.yml` | The launch form: account, partition, hours, memory; the cluster's name and where almasim is installed. |
| `submit.yml.erb` | The job: one CPU and the memory asked for. |
| `template/before.sh.erb` | Picks a free port and a fresh token, and the path the proxy forwards. |
| `template/script.sh.erb` | Starts `almasim serve --managed` with them. |
| `view.html.erb` | The Connect button: opens the console with the token once. |

## Installing

1. Set `cluster` and `almasim_setup` in `form.yml` for the site.
2. Copy this folder to `/var/www/ood/apps/sys/almasim/` (a system app), or
   to `~/ondemand/dev/almasim/` to try it as one user first.
3. The portal's `host_regex` must match the compute nodes' names, so its
   proxy forwards to them.

Nothing else is needed: the console runs as the user, with the user's
Slurm account and files, and stops when the session ends.
