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

singularity exec --nv  $image  scl enable rh-python38 bash << EOF > $LOG_LOC/$DATE.log >&1
python3.8 $SCRIPT_LOC/bin/processing_ppi_csv.py \
  $CSV_LOC
  $DATA_LOC > Oct17_ppi_csv_prepare_set1.log 2>&1

python3.8 $SCRIPT_LOC/bin/main.py \
  Oct17 \
  ppi \
  1 \
  $DATA_LOC \
  $OUTPUT_LOC \
  $TOKENIZER_LOC \
  --num_workers 32 \
  --num_hidden_layers 32\
  --batch_size 1 \
  --max_position_embeddings 4096 \
  --vocab_size 52k \
  --hidden_size 4096 \
  --intermediate_size 11008 > Oct17_ppi_pretrain_set1.log 2>&1

exit

EOF
exit 0
