#!/usr/bin/env bash
# Start a single-node Slurm cluster inside the CI container (MADIV-28 tier 2).
#
# munge, slurmctld and slurmd run in this container, with one partition "ci"
# covering its cores and memory. dask-jobqueue then submits real sbatch jobs
# whose workers run here, which exercises the pipeline's Slurm code paths
# (scheduling/cluster.py, worker environments, child-process isolation).
set -euo pipefail
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
if ! apt-get install -y -qq --no-install-recommends slurm-wlm munge > /tmp/apt-slurm.log 2>&1; then
    tail -40 /tmp/apt-slurm.log
    exit 1
fi

HOST="$(hostname)"
CPUS="$(nproc)"
MEM_MB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 - 1024 ))
mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
# A container has no systemd/dbus: slurmd's cgroup v2 plugin manages its own
# hierarchy instead (needs a writable /sys/fs/cgroup: --privileged --cgroupns=host).
# Slurm 23.11 has no CgroupPlugin=disabled.
printf '%s\n' 'CgroupPlugin=cgroup/v2' 'IgnoreSystemd=yes' > /etc/slurm/cgroup.conf
# The CI container shares the runner VM's cgroup namespace (--cgroupns=host),
# whose systemd already enabled the controllers. Without systemd, slurmd puts
# its stepd scope in <parent of its own cgroup>/system.slice: create that.
OWN_CGROUP="$(cut -d: -f3 /proc/self/cgroup | head -1)"
STEPD_PARENT="/sys/fs/cgroup$(dirname "$OWN_CGROUP")/system.slice"
echo "own cgroup: $OWN_CGROUP; slurmstepd scopes go under $STEPD_PARENT"
mkdir -p "$STEPD_PARENT"
cat "/sys/fs/cgroup$(dirname "$OWN_CGROUP")/cgroup.subtree_control" || true
# Node line with the real topology (sockets/cores/threads), memory reduced.
NODE_LINE="$(slurmd -C | head -1 | sed -E "s/RealMemory=[0-9]+/RealMemory=$MEM_MB/")"
cat > /etc/slurm/slurm.conf <<CONF
ClusterName=ci
SlurmctldHost=$HOST
SlurmUser=root
SlurmdUser=root
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
ProctrackType=proctrack/pgid
TaskPlugin=task/none
JobAcctGatherType=jobacct_gather/none
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
MpiDefault=none
ReturnToService=2
$NODE_LINE State=UNKNOWN
PartitionName=ci Nodes=$HOST Default=YES MaxTime=INFINITE State=UP
CONF

mkdir -p /etc/munge /run/munge /var/lib/munge /var/log/munge
[ -f /etc/munge/munge.key ] || dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none
chown -R munge:munge /etc/munge /run/munge /var/lib/munge /var/log/munge
chmod 0700 /etc/munge && chmod 0400 /etc/munge/munge.key
runuser -u munge -- munged --force

slurmctld
slurmd || { tail -20 /var/log/slurm/slurmd.log; exit 1; }
for _ in $(seq 60); do
    sinfo -h -o %T 2>/dev/null | grep -qE '^idle' && break
    sleep 2
done
sinfo
srun -N1 -n1 hostname
echo "Slurm cluster ready: partition ci, $CPUS CPUs, $MEM_MB MB on $HOST"
