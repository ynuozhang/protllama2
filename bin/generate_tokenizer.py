import sentencepiece as spm

spm_args = '--input=/data/rozen/home/e0833634/lama/protllama/notebooks/protein_corpus.txt ' \
           '--model_prefix=protein_50k ' \
           '--vocab_size=50000 ' \
           '--num_threads=40'

# Train the model
spm.SentencePieceTrainer.Train(spm_args)