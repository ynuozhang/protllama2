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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, \
    Timer, TQDMProgressBar, LearningRateMonitor, \
    GradientAccumulationScheduler, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler
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
    parser.add_argument('--num_attention_heads', type=int, default=32,
                        help='Number of attention heads, should be dividable by embedding dimension')
    parser.add_argument('--num_key_value_heads', type=int, default=32,
                        help='Number of attention heads, should be dividable by embedding dimension')
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
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=10,
                        help='Number of batches accumulated before calculating loss. Per GPU accumulation.')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Data parallelism strategies')
    parser.add_argument('--flash_attention', action='store_true',
                        help='Using flash attention 2 or not')
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
else:
    raise ValueError('Target not prepared for the training.')

hparam.train_dataloader_length = 5881 # 47054 // 8
if not os.path.exists(f'pretrain_protllama_{hparam.target}'):
    os.makedirs(f'pretrain_protllama_{hparam.target}')
training_log_path = str(f'pretrain_protllama_{hparam.target}_{hparam.date}/pl_logs/')
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
    monitor="test_loss",
    min_delta=0.0,
    patience=5,  # number of epoch with no improvement
    verbose=True,
    mode="min",
)
training_model_path = str(f'pretrain_protllama_{hparam.target}_{hparam.date}/pl_model_cache_{hparam.date}_attempt_{hparam.attempts}/')
if not os.path.exists(training_model_path):
    os.makedirs(training_model_path)
checkpoint_callback = ModelCheckpoint(
    dirpath=training_model_path,
    filename="{epoch}-{train_perplexity:.3f}-{val_perplexity:.3f}-{train_loss:.3f}-{val_loss:.3f}-{test_loss:.3f}_%s_%s_%s_%s" % (hparam.target, hparam.date, hparam.vocab_size, hparam.max_position_embeddings),
    save_top_k=hparam.save_top_k,
    verbose=True,
    monitor="test_loss",
    mode="min",
    every_n_epochs=1
)
lr_monitor = LearningRateMonitor(
    logging_interval='epoch'
)
if hparam.strategy == 'fsdp':
    policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
    strategy = FSDPStrategy(auto_wrap_policy=policy)
elif hparam.strategy == 'ddp':
    strategy = 'ddp'
elif hparam.strategy.contains('deepspeed'):
    strategy = 'deepspeed_stage_2' #Shard optimizer states and gradients, remains at speed parity with DDP whilst providing even more memory improvement
else:
    strategy = None

# accumulating grad batches
# till 5th epoch, it will accumulate every 10 batches. From 5th epoch
# till 9th epoch it will accumulate every 8 batches and after that no accumulation
# will happen. Note that you need to use zero-indexed epoch keys here
accumulator = GradientAccumulationScheduler(scheduling={0: 10, 2: 8, 4: 5, 6: 1})

profiler = AdvancedProfiler(dirpath=training_log_path+f'{hparam.target}_{hparam.date}_{hparam.attempts}/', filename="perf_logs")


trainer = Trainer(
    devices=hparam.devices,
    accelerator='gpu',
    strategy=strategy,
    #fast_dev_run=True,
    precision='bf16',
    #limit_train_batches=0.001, # 1% of train, shorten epoch length
    #limit_train_batches=0.001, # 1% of train, shorten epoch length
    #limit_val_batches=0.001, # 1% of val
    gradient_clip_val=1, # llama used gradient clipping=1, default is norm
    #accumulate_grad_batches=hparam.accumulate_grad_batches, # already set callbacks
    max_epochs=hparam.epoch,
    log_every_n_steps=100,
    logger=logger,
    default_root_dir=f'pretrain_protllama_{hparam.target}/pl_model_training_cache/',
    # max_epochs=1,
    # min_epochs=1,
    callbacks=[TQDMProgressBar(refresh_rate=10), accumulator,
               lr_monitor, checkpoint_callback,
               StochasticWeightAveraging(swa_lrs=1e-2)], # make the model generalize better, harder to stuck at local minimum
    deterministic=True,
    profiler=profiler, # profile execution time per function
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
timer.time_elapsed('test')
print(trainer.checkpoint_callback.best_model_path)
