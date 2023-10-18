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
                                          transformers_version=transformers.__version__,
                                          intermediate_size=self.hparam.intermediate_size,
                                          num_hidden_layers=self.hparam.num_hidden_layers,
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
            lr_schedulers = {
                "scheduler": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                            num_training_steps=self.hparam.epoch * self.hparam.train_dataset_length,
                                                            num_cycles=0.39758361765,
                                                            # number of waves in the cosine schedule - e.g. 0.5 for period 2 cos wave means take 0 to 1
                                                            last_epoch=-1  # index of the last epoch when resuming training
                                                            ),
                "name": 'learning_rate_logs'
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
        perplexity = torch.exp(loss_train.detach())  # Ensure outputs are on CPU

        # Accuracy computation
        # Shifting
        shift_logits = outputs[1][..., :-1, :].contiguous().argmax(dim=-1)  # Ensure outputs and argmax result are on CPU
        if verbose:
            print(shift_logits)

        # Assuming 'labels' is a key in batch containing true token IDs
        shift_labels = batch['labels'][..., 1:].contiguous()  # Move labels to CPU
        if verbose:
            print(shift_logits)

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

    def validation_step(self, batch, batch_nb: int):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        outputs = self.forward(**batch)
        loss_val = outputs[0]

        # Compute the perplexity
        perplexity = torch.exp(loss_val.detach())  # Ensure outputs are on CPU

        # Accuracy computation
        # Shifting
        shift_logits = outputs[1][..., :-1, :].contiguous().argmax(
            dim=-1)  # Ensure outputs and argmax result are on CPU

        # Assuming 'labels' is a key in batch containing true token IDs
        shift_labels = batch['labels'][..., 1:].contiguous()  # Move labels to CPU

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
        parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for Adam optimizer')
        parser.add_argument('--scheduler', type=str, default='linear', help='Learning rate scheduler, either linear '
                                                                            'or cosine')
        parser.add_argument('--epoch', type=int, default=50, help='number of epochs for the training')
        return parser


