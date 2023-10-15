#!/bin/sh

HOME_LOC=/home/a03-yzhang
DATA_LOC=$HOME_LOC/projects/protllama2_data/data
OUTPUT_LOC=$HOME_LOC/projects/protllama2_data/data/1million_example
mkdir $OUTPUT_LOC
SCRIPT_LOC=$HOME_LOC/projects/protllama2
TOKENIZER_LOC=$HOME_LOC/projects/protllama2_data/data/tokenizers/
LOG_LOC=$HOME_LOC/projects/protllama2_data/logs
mkdir $LOG_LOC
image=$HOME_LOC/containers/pure_centos.sif
DATE=$(date +%m_%d)

singularity exec --nv  $image  scl enable rh-python38 bash << EOF > $LOG_LOC/$DATE.log >&1

python3.8 $SCRIPT_LOC/bin/main.py \
  Oct13 \
  protein \
  2 \
  $DATA_LOC \
  $OUTPUT_LOC \
  $TOKENIZER_LOC \
  --num_workers 32 \
  --num_hidden_layers 32\
  --batch_size 8 \
  --max_position_embeddings 512 \
  --vocab_size 8k \
  --hidden_size 1024 \
  --intermediate_size 2752 > Oct13_pretrain_set2_.log 2>&1

exit

EOF
exit 0
