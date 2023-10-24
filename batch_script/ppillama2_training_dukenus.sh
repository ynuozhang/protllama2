#!/bin/sh
#PBS -N small
#PBS -q gpu
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=64:mem=400g:ngpus=2
#PBS -keo

cd $PBS_O_WORKDIR

HOME_LOC=/data/rozen/home/e0833634/lama
CSV_LOC=$HOME_LOC/data/ppi_8000
DATA_LOC=$HOME_LOC/data/ppi_8000
OUTPUT_LOC=$HOME_LOC/data/ppi_8000
mkdir $OUTPUT_LOC
SCRIPT_LOC=$HOME_LOC/protllama
TOKENIZER_LOC=$HOME_LOC/data/tokenizers/
LOG_LOC=$HOME_LOC/protllama/batch_script/logs
mkdir $LOG_LOC
image=/data/rozen/home/e0833634/distributed_containers/update_centos7_old_container_v2.sif
module load singularity
DATE=$(date +%m_%d)

singularity exec --nv  $image  scl enable rh-python38 bash << EOF > $LOG_LOC/$DATE.log 2>&1
printf $DATE
python3.8 $SCRIPT_LOC/bin/processing_ppi_csv.py \
  $CSV_LOC \
  $DATA_LOC > ${DATE}_dukenus_hpc_ppi_csv_prepare_set1.log 2>&1

printf pwd

python3.8 $SCRIPT_LOC/bin/main.py \
  ${DATE}_dukenus_hpc \
  ppi \
  3 \
  $DATA_LOC \
  $OUTPUT_LOC \
  $TOKENIZER_LOC \
  --num_workers 4 \
  --num_hidden_layers 30 \
  --num_attention_heads 20 \
  --num_key_value_heads 20 \
  --batch_size 1 \
  --max_position_embeddings 1024 \
  --vocab_size 32k \
  --hidden_size 1280 \
  --intermediate_size 3440 \
  --devices 1 \
  --accumulate_grad_batches 10 \
  --strategy ddp \
  --flash_attention True > ${DATE}_dukenus_hpc_ppi_pretrain_set6_full_ddp.log 2>&1

exit

EOF
exit 0
