"""Use pytorch-lightning to add customized head later (Oct 6)"""
import lightning.pytorch as pl
import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import sentencepiece as spm
from argparse import ArgumentParser
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_ratio = eta_ratio  # The ratio of minimum to maximum learning rate
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = (1 - self.eta_ratio) * cosine_decay + self.eta_ratio

        return [decayed_lr * base_lr for base_lr in self.base_lrs]


class pretrainLlama(pl.LightningModule):
    def __init__(self, hparam) -> None:
        super(pretrainLlama, self).__init__()
        self.save_hyperparameters()
        self.hparam = hparam  # need to contain epoch, target, date, learning rate, batch_size, num_frozen_epochs
        self.MODEL_CONFIGS = self.retrieve_config()
        self.model = None
        self.tokenizer = self.tokenizer_generation(self.hparam.tokenizer_path, self.hparam.target, self.hparam.vocab_size)

    @staticmethod
    def tokenizer_generation(tokenizer_path, target, vocab_size):
        if target == 'original':
            tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
            tokenizer.pad_token = tokenizer.unk_token
            return tokenizer
        elif target == 'protein' or target == 'ppi':
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
        elif self.hparam.target == 'protein' or self.hparam.target == 'ppi':
            config_dict = {
                'protllama2': LlamaConfig(max_position_embeddings=self.hparam.max_position_embeddings,  # maximum length
                                          hidden_size=self.hparam.hidden_size,
                                          num_attention_heads=self.hparam.num_attention_heads,
                                          num_key_value_heads=self.hparam.num_key_value_heads,
                                          transformers_version=transformers.__version__,
                                          intermediate_size=self.hparam.intermediate_size,
                                          num_hidden_layers=self.hparam.num_hidden_layers,
                                          _flash_attn_2_enabled=self.hparam.flash_attention,
                                          vocab_size=int(self.hparam.vocab_size.rstrip('k')) * 1000)}
            print(config_dict['protllama2'])
            return config_dict['protllama2']
        else:
            raise ValueError('Have not prepared dataset for this target')

    def configure_model(self):
        """start model building, can add customized classification head"""
        if self.model is not None:
            return
        self.model = LlamaForCausalLM(self.MODEL_CONFIGS)
        #print(self.model.lm_head.weight)

    def configure_optimizers(self):
        """set learning rates"""
        if self.hparam.scheduler == 'linear':
            parameters = self.model.parameters()
            optimizer = AdamW(parameters, lr=self.hparam.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
            lr_schedulers = {
                "scheduler": get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=2000,
                                                            num_training_steps=self.hparam.epoch * self.hparam.train_dataset_length),
                "name": 'learning_rate_logs'
            }
            return [optimizer], [lr_schedulers]
        elif self.hparam.scheduler == 'cosine':
            """llama behavior, end learning rate matches 10% of the maximum learning rate
                hard-coded to be 10% first
            """
            parameters = self.model.parameters()
            optimizer = AdamW(parameters, lr=self.hparam.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
            schedulers = CosineAnnealingWithWarmup(optimizer, warmup_steps=2000, eta_ratio=0.1, total_steps=self.hparam.epoch * self.hparam.train_dataloader_length)
            lr_schedulers = {
                "scheduler": schedulers,
                "name": 'learning_rate_logs',
                "interval": 'step', # The scheduler updates the learning rate at every step (not epoch)
                'frequency': 1 # The scheduler updates the learning rate after every batch
            }

            return [optimizer], [lr_schedulers]
        else:
            raise ValueError('You need to specify a scheduler first. Default is linear')

    def forward(self, **inputs):
        """ Pytorch forward function
        Returns:
        dict with model outputs (loss, logits, hidden layer, attention)
        """
        return self.model(**inputs)

    def training_step(self, batch, batch_nb: int, verbose=False):
        if verbose:
            print(batch.keys())
            print(f"training_step input_ids shape: {batch['input_ids'].shape}")
        outputs = self.forward(**batch)
        loss_train = outputs[0]

        # Compute the perplexity
        perplexity = torch.exp(loss_train)  # Ensure outputs are on CPU

        # Accuracy computation
        # Shifting
        shift_logits = outputs[1][..., :-1, :].argmax(dim=-1)  # Ensure outputs and argmax result are on CPU

        # Assuming 'labels' is a key in batch containing true token IDs
        shift_labels = batch['labels'][..., 1:]  # Move labels to CPU

        non_padding_mask = shift_labels != -100

        # Compare predictions to true labels, but only for non-padding tokens
        acc_train = ((shift_logits == shift_labels) & non_padding_mask).sum().item() / non_padding_mask.sum().item()
        if verbose:
            print(acc_train)

        # Log
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_perplexity', perplexity.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('train_accuracy', acc_train, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss_train

    def validation_step(self, batch, batch_nb: int, verbose=False):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        if verbose:
            print(batch.keys())
            print(f"validatation_step input_ids shape: {batch['input_ids'].shape}")

        outputs = self.forward(**batch)
        loss_val = outputs[0]

        # Compute the perplexity
        perplexity = torch.exp(loss_val)  # Ensure outputs are on CPU

        # Accuracy computation
        # Shifting
        shift_logits = outputs[1][..., :-1, :].argmax(
            dim=-1)  # Ensure outputs and argmax result are on CPU

        # Assuming 'labels' is a key in batch containing true token IDs
        shift_labels = batch['labels'][..., 1:]  # Move labels to CPU

        non_padding_mask = shift_labels != -100

        # Compare predictions to true labels, but only for non-padding tokens
        acc_val = ((shift_logits == shift_labels) & non_padding_mask).sum().item() / non_padding_mask.sum().item()

        # Log
        self.log('val_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_accuracy', acc_val, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss_val

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser):
        """parser for hyperparameters"""
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')
        parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler, either linear '
                                                                            'or cosine')
        parser.add_argument('--epoch', type=int, default=30, help='number of epochs for the training')
        return parser


