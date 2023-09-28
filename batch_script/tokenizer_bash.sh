#!/bin/sh
#PBS -N tokenizer_generation
#PBS -q super
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=40:mem=200gb
#PBS -keo

HOME_LOC=/data/rozen/home/e0833634/lama/protllama/
cd $PBS_O_WORKDIR
image=/data/rozen/home/e0833634/distributed_containers/update_centos7_old_container_v2.sif
module load singularity

for vocab_size in 8 32 40 50; do

singularity exec --nv  $image  scl enable rh-python38 bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

python3.8 /data/rozen/home/e0833634/lama/protllama/bin/generate_tokenizer.py $vocab_size > Sept21_generate_protein_tokenizer_${vocab_size}k.log 2>&1

exit

EOF
done