import argparse
import os
from .model import *
from .data import PretrainDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# set wandb offline on HPC
os.environ['WANDB_MODE'] = "offline"

def parse_args():
    parser = argparse.ArgumentParser(description='pre-training protllama2')
    parser.add_argument('date', type=str,
                        help='Working as a namespace for every file generated')
    parser.add_argument('target', type=str,
                        help='Working as a namespace for every file generated')
    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help='Maximum length of the input')
    parser.add_argument('--hidden_size', type=int, default=640,
                        help='Embedding dimension')
    parser.add_argument('--intermediate_size', type=int, default=1720,
                        help='MLP FFNN intermediate feature dimension')
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser = pretrainLlama.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


hparam = parse_args()
dm = PretrainDataset(target=hparam.target,
                     max_sequence_length=hparam.max_position_embeddings)
# make sure dataset has "training" key
hparam.train_dataset_length = len(dm.dataset['train'])
training_log_path = str('protllama/pl_logs/')
if not os.path.exists(training_log_path):
    os.makedirs(training_log_path)
logger = WandbLogger(project="protllama2",
                     name="%s_%s_pre-training_log" % (hparam.target, hparam.date), #display on the web
                     save_dir='protllama/pl_logs/',
                     job_type='model-training',
                     group='pretrain_protllama2',
                     id='version_%s' % str(1))
seed_everything(42)
model = pretrainLlama(hparam)
early_stop_callback = EarlyStopping(
    monitor="loss",
    min_delta=0.0,
    patience=0,  # number of epoch with no improvement
    verbose=True,
    mode="min",
)
training_model_path = str('protllama/pl_model_cache/')
if not os.path.exists(training_model_path):
    os.makedirs(training_model_path)
checkpoint_callback = ModelCheckpoint(
    dirpath=training_model_path,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-%s_%s" % (hparam.target, hparam.date),
    save_top_k=hparam.save_top_k,
    verbose=True,
    monitor="loss",
    mode="min",
)
lr_monitor = LearningRateMonitor(
    logging_interval='epoch'
)
trainer = Trainer(
    devices=1,
    accelerator='gpu',
    limit_train_batches=3,
    max_epochs=hparam.epoch,
    logger=logger,
    # max_epochs=1,
    # min_epochs=1,
    callbacks=[TQDMProgressBar(refresh_rate=10), lr_monitor],
    deterministic=True,
    enable_model_summary=True
)

# automatic garbage collection
import gc

gc.collect()
trainer.fit(model, datamodule=dm)
print(trainer.checkpoint_callback.best_model_path)
