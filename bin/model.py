"""Use pytorch-lightning to add customized head later (Oct 6)"""
import pytorch_lightning as pl
import pickle
from tokenizers import Tokenizer
from datasets import Dataset
import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
from functools import partial
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import os
import logging as log
import glob
from argparse import ArgumentParser


class pretrainDataset(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 1,
                 target: str = 'original',
                 max_sequence_length: int = 1024):
        super().__init__()
        self.dataset = self.retrieve_data(target)
        self.tokenizer = self.tokenizer_generation(target)
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size

    @staticmethod
    def retrieve_data(target):
        """ return transformers DATASET object"""
        if target == 'original':
            with open('/data/rozen/home/e0833634/lama/protllama/original_lama.pkl', 'rb') as f:
                loaded_data_lama = pickle.load(f)
            return loaded_data_lama
        elif target == 'protein':
            with open('/data/rozen/home/e0833634/lama/protllama/uniprot_dataset.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
            return loaded_data
        else:
            raise ValueError('Have not prepared dataset for this target')

    @staticmethod
    def tokenizer_generation(target):
        if target == 'original':
            tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
            tokenizer.pad_token = tokenizer.unk_token
            return tokenizer
        elif target == 'protein':
            path = '../../protgpt2/'
            tokenizer = Tokenizer.from_file(path + 'tokenizer.json')
            # Set bos_token_id and eos_token_id to 0
            tokenizer.bos_token_id = 0
            tokenizer.eos_token_id = 0
            # Confirming
            assert tokenizer.bos_token_id == 0
            assert tokenizer.eos_token_id == 0
            return tokenizer
        else:
            raise ValueError('Have not prepared tokenizer for this target')

    def train_dataloader(self):
        if 'train' in self.dataset.keys():
            train_ds = self.dataset['train']
            return DataLoader(train_ds, batch_size=self.batch_size, shuffle=False,
                              drop_last=True,
                              collate_fn=partial(self.tokenize_batch))
        elif 'train' not in self.dataset.keys():
            raise ValueError('Did not detect training dataset')

    def val_dataloader(self):
        if 'valid' in self.dataset.keys():
            val_ds = self.dataset['valid']
            return DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                              drop_last=False,
                              collate_fn=partial(self.tokenize_batch))
        else:
            valid_dict = {'text': [
                "Since it was initiated by the Brazil workers' party~\cite{wainwright2003making} in the 90s, Participatory budgeting (PB)~\cite{cabannes2004participatory}"]}
            val_ds = Dataset.from_dict(valid_dict)
            return DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                              drop_last=False,
                              collate_fn=partial(self.tokenize_batch))

    def tokenize_batch(self, batch):
        texts = [sample['text'] for sample in batch]
        data = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True,
                              max_length=self.max_sequence_length)
        data['labels'] = data['input_ids'].clone()
        return data


class pretrainLlama(pl.LightningModule):
    def __init__(self, hparam) -> None:
        super(pretrainLlama, self).__init__()
        self.save_hyperparameters()
        self.hparam = hparam  # need to contain epoch, target, date, learning rate, batch_size, num_frozen_epochs
        self.tokenizer = pretrainDataset(self.hparam.target).tokenizer
        self.MODEL_CONFIGS = self.retrieve_config()
        self.__build_model()

    def retrieve_config(self):
        """ return transformers DATASET object"""
        if self.hparam.target == 'original':
            config_dict = {'7b': LlamaConfig(max_position_embeddings=self.hparam.max_position_embeddings,
                                             hidden_size=self.hparam.hidden_size,
                                             intermediate_size=self.hparam.intermediate_size)}
            return config_dict
        elif self.hparam.target == 'protein':
            config_dict = {
                'protllama2': LlamaConfig(max_position_embeddings=self.hparam.max_position_embeddings,  # maximum length
                                          hidden_size=self.hparam.hidden_size,
                                          bos_token_id=self.tokenizer.bos_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          transformers_version=transformers.__version__,
                                          intermediate_size=self.hparam.intermediate_size,
                                          vocab_size=50257)}
            return config_dict
        else:
            raise ValueError('Have not prepared dataset for this target')

    def check_point_check(self):
        os.mkdir('/data/rozen/home/e0833634/transformerDENV/result/model_cache/%s' % self.hparam.date)
        # try to load from checkpoints
        cache_list = glob.glob(
            '/data/rozen/home/e0833634/transformerDENV/result/model_cache/%s/*' % self.hparam.date)
        try:
            # get the most recently saved checkpoint
            cache_fname = max(cache_list, key=os.path.getctime)
            log.info(f'\n-- model at check point is loaded successfully')
            return FtESM1b.load_from_checkpoint(cache_fname)
        except FileNotFoundError:
            log.info(f'\n-- Did not found any checkpoints.')

    def __build_model(self) -> None:
        """start model building, can add customized classification head"""
        config = '7b'
        config = self.MODEL_CONFIGS[config]
        self.model = LlamaForCausalLM(config)
        print(self.model.lm_head.weight)

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

    def training_step(self, batch, batch_nb: int):
        for k, v in batch.items():
            batch[k] = v
        outputs = self.forward(**batch)
        loss_train = outputs[0]
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_nb: int):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        for k, v in batch.items():
            batch[k] = v
        outputs = self.forward(**batch)
        loss_val = outputs[0]
        self.log('val_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_val

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser):
        """parser for hyperparameters"""
        parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for Adam optimizer')
        parser.add_argument('--scheduler', type=str, default='linear', help='Learning rate scheduler, either linear '
                                                                            'or cosine')
        parser.add_argument('--epoch', type=int, default=4, help='number of epochs for the training')
        parser.add_argument('--batch_size', type=int, default=2, help='Batch sizes, sequence number per batch')
        return parser


if __name__ == '__main__':
    #dm = pretrainDataset()
    #dm.setup("fit")
    #print(next(iter(dm.train_dataloader())))
    dm = pretrainDataset(batch_size=1)
    # make sure dataset has "training" key
    print(type(dm.dataset['train']))
    print(len(dm.dataset['train']))
    print(len(dm.dataset['train']))

