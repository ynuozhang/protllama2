import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()
import argparse
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import transformers
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import os
import logging as log
import glob
from argparse import ArgumentParser
import torch
import numpy as np
import pytorch_lightning as pl
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import sentencepiece as spm
import pickle
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from torch.nn.utils.rnn import pad_sequence


dataset_path = f'/data/rozen/home/e0833634/lama/protllama/batch_script/uniref50_random90split_8k_512_first_1million_dataset.hf'
dataset = load_from_disk(dataset_path)


small_dataset_dict = DatasetDict({
    'train': dataset['train'].select(range(10)),
    'valid': dataset['valid'].select(range(10))
})


with open('/data/rozen/home/e0833634/lama/protllama/batch_script/train_intermediate_checkpoint_batches_1000000.pkl', 'rb') as f:
    batch_indices_train = pickle.load(f)
batch_indices_train = batch_indices_train[:10]

with open('/data/rozen/home/e0833634/lama/protllama/batch_script/valid_intermediate_checkpoint_batches_1000000.pkl', 'rb') as f:
    batch_indices_val = pickle.load(f)
batch_indices_val = batch_indices_val[:10]

from torch.utils.data import Dataset, DataLoader
import torch

class DynamicBatchingDataset(Dataset):
    def __init__(self, dataset_dict, batch_indices):
        print('Initializing dataset...')
        self.dataset_dict = dataset_dict
        self.batch_indices = batch_indices  # This is mainly for informational purposes, if needed.

    def __len__(self):
        return len(self.dataset_dict['attention_mask'])  # Assuming each entry in dataset_dict represents a batch

    def __getitem__(self, idx):
        #batch_idx = self.batch_indices[idx]
        # Directly retrieve the batch using the index
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(f"Process 0 is fetching index: {idx}")
            elif torch.distributed.get_rank() == 1:
                print(f"Process 1 is fetching index: {idx}")
        """returns [seq_number, token_length], return one batch at a time"""
        attention_mask = torch.tensor(self.dataset_dict['attention_mask'][idx])
        input_ids = torch.tensor(self.dataset_dict['input_ids'][idx])
        label = torch.tensor(self.dataset_dict['labels'][idx])

        return {
            'attention_mask': attention_mask,
            'input_ids': input_ids,
            'labels': label
        }

    @staticmethod
    def custom_collate_fn(batch):
        """Return the first element of the batch because each element is already a batch.
            (prevent auto batching from pytorch DataLoader, otherwise, the batch_size=1 will add another dimension during data retrieval)
        """
        return batch[1]

    @staticmethod
    def dynamic_padding_collate_fn(batch):
        # Extract sequences from the batch
        attention_masks = [item['attention_mask'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Flatten the sequences and then pad them
        attention_masks_flat = [sequence for batch_item in attention_masks for sequence in batch_item]
        input_ids_flat = [sequence for batch_item in input_ids for sequence in batch_item]
        labels_flat = [sequence for batch_item in labels for sequence in batch_item]

        # Pad sequences
        attention_masks_padded = pad_sequence(attention_masks_flat, batch_first=True)
        input_ids_padded = pad_sequence(input_ids_flat, batch_first=True)
        labels_padded = pad_sequence(labels_flat, batch_first=True, padding_value=-100)

        return {
            'attention_mask': attention_masks_padded,
            'input_ids': input_ids_padded,
            'labels': labels_padded
        }

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dict, batch_indices_train, batch_indices_val, batch_size=1):
        super().__init__()
        self.dataset_dict = dataset_dict
        self.batch_indices_train = batch_indices_train
        self.batch_indices_val = batch_indices_val
        self.batch_size = batch_size

        self.tokenizer = self.tokenizer_generation('protein', '8k')

    @staticmethod
    def tokenizer_generation(target, vocab_size):
        if target == 'original':
            tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
            tokenizer.pad_token = tokenizer.unk_token
            return tokenizer
        elif target == 'protein':
            tokenizer_path = '/data/rozen/home/e0833634/lama/protllama/batch_script/'
            tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path + "protein_%s.model" % (vocab_size))
            return tokenizer
        else:
            raise ValueError('Have not prepared tokenizer for this target')

    def prepare_data(self):
        # Possibly download data, set transforms, etc.
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = DynamicBatchingDataset(self.dataset_dict['train'], self.batch_indices_train)
        # Repeat similar steps for validation and test datasets if needed
        self.val_dataset = DynamicBatchingDataset(self.dataset_dict['valid'], self.batch_indices_val)

    def train_dataloader(self):
        print("dataloader created...")
        d = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=DynamicBatchingDataset.dynamic_padding_collate_fn)
        if self.trainer.global_rank == 0:
            for idx, batch in enumerate(d):
                print(idx, batch)
        return d

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=DynamicBatchingDataset.dynamic_padding_collate_fn)

class pretrainLlama(pl.LightningModule):
    def __init__(self, hparam) -> None:
        super(pretrainLlama, self).__init__()
        self.save_hyperparameters()
        self.hparam = hparam  # need to contain epoch, target, date, learning rate, batch_size, num_frozen_epochs
        self.MODEL_CONFIGS = self.retrieve_config()
        self.__build_model()
        self.tokenizer = self.tokenizer_generation('protein', '8k')

    @staticmethod
    def tokenizer_generation(target, vocab_size):
        if target == 'original':
            tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
            tokenizer.pad_token = tokenizer.unk_token
            return tokenizer
        elif target == 'protein':
            tokenizer_path = '/data/rozen/home/e0833634/lama/protllama/batch_script/'
            tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path + "protein_%s.model" % (vocab_size))
            return tokenizer
        else:
            raise ValueError('Have not prepared tokenizer for this target')

    def retrieve_config(self):
        """ return transformers DATASET object"""
        if self.hparam.target == 'original':
            config_dict = {'7b': LlamaConfig(max_position_embeddings=self.hparam.max_position_embeddings,
                                             hidden_size=self.hparam.hidden_size,
                                             intermediate_size=self.hparam.intermediate_size)}
            return config_dict['7b']
        elif self.hparam.target == 'protein':
            config_dict = {
                'protllama2': LlamaConfig(max_position_embeddings=self.hparam.max_position_embeddings,  # maximum length
                                          hidden_size=self.hparam.hidden_size,
                                          transformers_version=transformers.__version__,
                                          intermediate_size=self.hparam.intermediate_size,
                                          vocab_size=int(self.hparam.vocab_size.rstrip('k')) * 1000)}
            #print(config_dict['protllama2'])
            return config_dict['protllama2']
        else:
            raise ValueError('Have not prepared dataset for this target')

    def __build_model(self) -> None:
        """start model building, can add customized classification head"""
        self.model = LlamaForCausalLM(self.MODEL_CONFIGS)
        #print(self.model.lm_head.weight)

    def configure_optimizers(self):
        """set learning rates"""
        if self.hparam.scheduler == 'linear':
            parameters = self.model.parameters()
            optimizer = AdamW(parameters, lr=self.hparam.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
            lr_schedulers = {
                "scheduler": get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=100,
                                                            num_training_steps=self.hparam.epoch * self.hparam.train_dataset_length),
                "name": 'learning_rate_logs'
            }
            return [optimizer], [lr_schedulers]

    def forward(self, **inputs):
        """ Pytorch forward function
        Returns:
        dict with model outputs (loss, logits, hidden layer, attention)
        """
        return self.model(**inputs)

    def training_step(self, batch, batch_nb: int, verbose=False):
        if torch.distributed.is_initialized():
            print(f"Process {torch.distributed.get_rank()} starting training step")
        outputs = self.forward(**batch)
        loss_train = outputs[0]

        # Compute the perplexity
        perplexity = torch.exp(outputs[0].cpu())  # Ensure outputs are on CPU

        # Accuracy computation
        # Shifting
        shift_logits = outputs[1][..., :-1, :].contiguous().argmax(
            dim=-1).cpu()  # Ensure outputs and argmax result are on CPU

        # Assuming 'labels' is a key in batch containing true token IDs
        shift_labels = batch['labels'][..., 1:].contiguous().cpu()  # Move labels to CPU

        non_padding_mask = shift_labels != -100
        # Compare predictions to true labels, but only for non-padding tokens
        acc_train = ((shift_logits == shift_labels) & non_padding_mask).sum().item() / non_padding_mask.sum().item()

        print('train', loss_train, perplexity, acc_train)

        # Log
        self.log('train_loss', loss_train, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_accuracy', acc_train, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss_train

    def validation_step(self, batch, batch_nb: int, verbose=False):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        outputs = self.forward(**batch)
        loss_val = outputs[0].cpu()

        # Compute the perplexity
        perplexity = torch.exp(loss_val)  # Ensure outputs are on CPU

        # Accuracy computation
        # Shifting
        shift_logits = outputs[1][..., :-1, :].contiguous().argmax(
            dim=-1).cpu()  # Ensure outputs and argmax result are on CPU

        # Assuming 'labels' is a key in batch containing true token IDs
        shift_labels = batch['labels'][..., 1:].contiguous().cpu()  # Move labels to CPU

        non_padding_mask = shift_labels != -100

        # Compare predictions to true labels, but only for non-padding tokens
        acc_val = ((shift_logits == shift_labels) & non_padding_mask).sum().item() / non_padding_mask.sum().item()

        print('val', loss_val, perplexity, acc_val)
        # Log
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss_val


from types import SimpleNamespace

hparam = SimpleNamespace(
    date='Oct_11',
    target='protein',
    max_position_embeddings=512,
    vocab_size='8k',
    hidden_size=640,
    intermediate_size=1720,
    save_top_k=1,
    scheduler='linear',
    learning_rate=3e-4,
    epoch=1
    #... add all other arguments similarly
)

data_module = CustomDataModule(dataset_dict=small_dataset_dict,
                               batch_indices_train=batch_indices_train,
                               batch_indices_val=batch_indices_val,
                               batch_size=2)
#dm = PretrainDataset(target=hparam.target,
                     #max_sequence_length=hparam.max_position_embeddings)
hparam.train_dataset_length = 10
training_log_path = str('protllama/pl_logs/')
if not os.path.exists(training_log_path):
    os.makedirs(training_log_path)
logger = WandbLogger(project="protllama2",
                     name=f"{hparam.target}_{hparam.date}_pre-training_log", #display on the web
                     save_dir='protllama/pl_logs/',
                     job_type='model-training',
                     group=f'pretrain_protllama2_{hparam.vocab_size}_{hparam.max_position_embeddings}',
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
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-%s_%s_%s_%s" % (hparam.target, hparam.date, hparam.vocab_size, hparam.max_position_embeddings),
    save_top_k=hparam.save_top_k,
    verbose=True,
    monitor="val_loss",
    mode="min",
)
lr_monitor = LearningRateMonitor(
    logging_interval='epoch'
)
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Or another interface if not eth0

print("About to initialize the Trainer...")
trainer = Trainer(
    devices=2,
    accelerator='gpu',
    strategy='ddp_notebook',
    num_nodes=1,
    fast_dev_run=True,
    limit_train_batches=2,
    max_epochs=1,
    logger=logger,
    # max_epochs=1,
    # min_epochs=1,
    #callbacks=[TQDMProgressBar(refresh_rate=10), lr_monitor],
    #deterministic=True,
    enable_model_summary=True
)
print("Trainer initialized.")
torch.set_float32_matmul_precision('medium')
# automatic garbage collection
import gc
gc.collect()

if torch.distributed.is_initialized():
    print(f"Process {torch.distributed.get_rank()} initialized")

trainer.fit(model, datamodule=data_module)
