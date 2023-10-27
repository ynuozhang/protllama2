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
image=$HOME_LOC/containers/update_centos7_old_container_v3.sif
DATE=$(date +%m_%d)

singularity exec --nv  $image  scl enable rh-python38 bash << EOF > $LOG_LOC/$DATE.log 2>&1

python3.8 $SCRIPT_LOC/bin/main.py \
  $DATE \
  ppi \
  1 \
  $DATA_LOC \
  $OUTPUT_LOC \
  $TOKENIZER_LOC \
  --num_workers 8 \
  --num_hidden_layers 30 \
  --num_attention_heads 20 \
  --num_key_value_heads 20 \
  --batch_size 1 \
  --max_position_embeddings 2048 \
  --vocab_size 10k \
  --hidden_size 1280 \
  --intermediate_size 3440 \
  --devices 8 \
  --accumulate_grad_batches 10 \
  --strategy ddp \
  --flash_attention > ${DATE}_ppi_pretrain_set1_full_ddp.log 2>&1

exit

EOF
exit 0
