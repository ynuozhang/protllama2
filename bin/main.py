import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'
from model import *
from data import PretrainDataset
from ppi_data import PretrainPPIDataset
import torch
import torch.nn as nn
from lightning.pytorch.strategies import FSDPStrategy
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import time

# set wandb offline on HPC
#os.environ['WANDB_MODE'] = "offline"

def parse_args():
    parser = argparse.ArgumentParser(description='pre-training protllama2')
    parser.add_argument('date', type=str,
                        help='Working as a namespace for every file generated')
    parser.add_argument('target', type=str,
                        help='Working as a namespace for every file generated')
    parser.add_argument('attempts', type=str,
                        help='Number of attempts tried for training')
    parser.add_argument('input_dataset_path', type=str,
                        help='Path for the dataset, should be Huggingface Datasets object')
    parser.add_argument('output_dataset_path', type=str,
                        help='Path for the dataset, should be Huggingface Datasets object')
    parser.add_argument('tokenizer_path', type=str,
                        help='Path for the tokenizers')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for the DataLoader to load datasets')
    parser.add_argument('--num_hidden_layers', type=int, default=32,
                        help='Number of hidden layers')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch sizes, sequence number per batch, used for DDP parallelism, number=no of GPUs')
    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help='Maximum length of the input')
    parser.add_argument('--vocab_size', type=str, default='8k',
                        help='Vocab size used to generate sentencepiece tokenizer')
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='Embedding dimension')
    parser.add_argument('--intermediate_size', type=int, default=2752,
                        help='MLP FFNN intermediate feature dimension')

    parser.add_argument(
        "--save_top_k",
        default=3,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser = pretrainLlama.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


hparam = parse_args()
if hparam.target == 'protein':
    dm = PretrainDataset(
        input_dataset_path=hparam.input_dataset_path,
        output_dataset_path=hparam.output_dataset_path,
        tokenizer_path=hparam.tokenizer_path,
        num_workers=hparam.num_workers,
        batch_size=hparam.batch_size,
        vocab_size=hparam.vocab_size,
        target=hparam.target,
        max_sequence_length=hparam.max_position_embeddings,
        )
elif hparam.target == 'ppi':
    dm = PretrainPPIDataset(
        input_dataset_path=hparam.input_dataset_path,
        output_dataset_path=hparam.output_dataset_path,
        tokenizer_path=hparam.tokenizer_path,
        num_workers=hparam.num_workers,
        batch_size=hparam.batch_size,
        vocab_size=hparam.vocab_size,
        target=hparam.target,
        max_sequence_length=hparam.max_position_embeddings,
    )

hparam.train_dataset_length = len(dm.dataset['train'])
training_log_path = str(f'pretrain_protllama_{hparam.target}/pl_logs/')
if not os.path.exists(training_log_path):
    os.makedirs(training_log_path)
logger = WandbLogger(project=f"pretrain_protllama_{hparam.target}",
                     name=f"{hparam.target}_{hparam.date}_{hparam.vocab_size}_pre-training_log", #display on the web
                     save_dir=f'pretrain_protllama_{hparam.target}/pl_logs/',
                     job_type='model-training',
                     group=f'pretrain_protllama2_{hparam.vocab_size}_{hparam.max_position_embeddings}',
                     id=f'version_{hparam.attempts}')

seed_everything(42)

model = pretrainLlama(hparam)
early_stop_callback = EarlyStopping(
    monitor="val_perplexity",
    min_delta=0.0,
    patience=5,  # number of epoch with no improvement
    verbose=True,
    mode="min",
)
training_model_path = str(f'pretrain_protllama_{hparam.target}/pl_model_cache/')
if not os.path.exists(training_model_path):
    os.makedirs(training_model_path)
checkpoint_callback = ModelCheckpoint(
    dirpath=training_model_path,
    filename="{epoch}-{train_perplexity:.3f}-{val_perplexity:.3f}-%s_%s_%s_%s" % (hparam.target, hparam.date, hparam.vocab_size, hparam.max_position_embeddings),
    save_top_k=-1,
    verbose=True,
    monitor="val_perplexity",
    mode="min",
    every_n_epochs=1
)
lr_monitor = LearningRateMonitor(
    logging_interval='epoch'
)
policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
strategy = FSDPStrategy(auto_wrap_policy=policy)
trainer = Trainer(
    devices=8,
    accelerator='cuda',
    strategy=strategy,
    #fast_dev_run=True,
    precision=16,
    #limit_train_batches=3,
    max_epochs=hparam.epoch,
    logger=logger,
    default_root_dir=f'pretrain_protllama_{hparam.target}/pl_model_training_cache/',
    # max_epochs=1,
    # min_epochs=1,
    callbacks=[TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback],
    deterministic=True,
    enable_model_summary=True
)
timer = Timer()
# automatic garbage collection
import gc
gc.collect()
trainer.fit(model, datamodule=dm)
trainer.print(torch.cuda.memory_summary())
timer.time_elapsed('train')
timer.time_elapsed('validate')
print(trainer.checkpoint_callback.best_model_path)
