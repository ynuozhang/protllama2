#!/bin/sh

HOME_LOC=/home/a03-yzhang
CSV_LOC=$HOME_LOC/projects/protllama2_data/data/ppi
DATA_LOC=$HOME_LOC/projects/protllama2_data/data/ppi/ppi_8000
OUTPUT_LOC=$HOME_LOC/projects/protllama2_output/ppi_8000
mkdir $OUTPUT_LOC
SCRIPT_LOC=$HOME_LOC/projects/protllama2
TOKENIZER_LOC=$HOME_LOC/projects/protllama2_data/data/tokenizers/
LOG_LOC=$HOME_LOC/projects/protllama2_data/logs
mkdir $LOG_LOC
image=$HOME_LOC/containers/pure_centos.sif
DATE=$(date +%m_%d)

singularity exec --nv  $image  scl enable rh-python38 bash << EOF > $LOG_LOC/$DATE.log 2>&1

python3.8 $SCRIPT_LOC/bin/main.py \
  $DATE \
  ppi \
  4 \
  $DATA_LOC \
  $OUTPUT_LOC \
  $TOKENIZER_LOC \
  --num_workers 8 \
  --num_hidden_layers 28\
  --batch_size 1 \
  --max_position_embeddings 4096 \
  --vocab_size 50k \
  --hidden_size 2048 \
  --intermediate_size 6880 > ${DATE}_ppi_pretrain_set4_full_fsdp.log 2>&1

exit

EOF
exit 0
