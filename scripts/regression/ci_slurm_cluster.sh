#!/usr/bin/env bash
# Start a single-node Slurm cluster inside the CI container (MADIV-28 tier 2).
#
# munge, slurmctld and slurmd run in this container, with one partition "ci"
# covering its cores and memory. dask-jobqueue then submits real sbatch jobs
# whose workers run here, which exercises the pipeline's Slurm code paths
# (scheduling/cluster.py, worker environments, child-process isolation).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends slurm-wlm munge > /dev/null

HOST="$(hostname)"
CPUS="$(nproc)"
MEM_MB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 - 1024 ))
mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
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
NodeName=$HOST CPUs=$CPUS RealMemory=$MEM_MB State=UNKNOWN
PartitionName=ci Nodes=$HOST Default=YES MaxTime=INFINITE State=UP
CONF

mkdir -p /etc/munge /run/munge /var/lib/munge /var/log/munge
[ -f /etc/munge/munge.key ] || dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none
chown -R munge:munge /etc/munge /run/munge /var/lib/munge /var/log/munge
chmod 0700 /etc/munge && chmod 0400 /etc/munge/munge.key
runuser -u munge -- munged --force

slurmctld
slurmd
for _ in $(seq 60); do
    sinfo -h -o %T 2>/dev/null | grep -qE '^idle' && break
    sleep 2
done
sinfo
srun -N1 -n1 hostname
echo "Slurm cluster ready: partition ci, $CPUS CPUs, $MEM_MB MB on $HOST"
