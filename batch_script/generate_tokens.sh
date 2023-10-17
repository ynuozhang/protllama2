#!/bin/sh
#PBS -N tokens
#PBS -q gpu
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=32:mem=300g:ngpus=1:vnode=computeg02
#PBS -keo

HOME_LOC=/data/rozen/home/e0833634/lama/protllama/
cd $PBS_O_WORKDIR
image=/data/rozen/home/e0833634/distributed_containers/update_centos7_old_container_v2.sif
module load singularity

singularity exec --nv  $image  scl enable rh-python38 bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

python3.8 /data/rozen/home/e0833634/lama/protllama/bin/ppi_data.py > Oct17_test_ppi_generate_tokens_8k.log 2>&1

exit

EOF
