import sentencepiece as spm
import sys
import datetime

vocab_size = sys.argv[1]

spm_args = f'--input=/data/rozen/home/e0833634/lama/data/swiss_2023_9/protein_corpus.txt ' \
           f'--model_prefix=protein_{vocab_size}k ' \
           f'--vocab_size={vocab_size}000 ' \
           '--num_threads=40'

sys.stdout.write(str(datetime.datetime.now()) + ' | ')
sys.stdout.write(f'Finish generating tokenizer for vocab size {vocab_size}k' + '\n')
sys.stdout.flush()
# Train the model
spm.SentencePieceTrainer.Train(spm_args)

