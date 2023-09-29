import pytorch_lightning as pl
import torch
import pickle
from tokenizers import Tokenizer
from datasets import Dataset
from functools import partial
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader
import sentencepiece as spm


class ProteinTokenizer(object):
    # inspired by https://github.com/facebookresearch/esm/blob/main/esm/data.py
    # in handling batches and tokenization
    def __init__(self, vocab_size):
        path = '/data/rozen/home/e0833634/lama/protllama/batch_script/'
        self.tokenizer = spm.SentencePieceProcessor(model_file=path+'protein_'+vocab_size+'.model')
        self.padding_idx = self.tokenizer.pad_id()
        self.unk_idx = self.tokenizer.unk_id()
        # original llama doesn't have <mask> token
        self.all_special_tokens = ['<s>', '</s>', '<pad>', '<unk>']
        self.unique_no_split_tokens = ['<s>', '</s>']

    def __len__(self):
        return self.tokenizer.get_piece_size()

    def get_idx(self, tok):
        return self.tokenizer.piece_to_id(tok)

    def get_tok(self, idx):
        return self.tokenizer.id_to_piece(idx)

    def to_dict(self):
        return {self.get_tok(i): i for i in range(len(self))}

    def get_batch_converter(self, max_sequence_length: int = None):
        return TokenizeBatch(self, max_sequence_length)

    def tokenize(self, text, **kwargs):
        """the sentencepiece model must be trained already
        output: split strings automatically completed by sentencepiece
        """
        return self.tokenizer.encode(text, out_type=str)

    def encode(self, text):
        """same requirement as tokenize
        output: tokens instead of strings
        """
        return self.tokenizer.encode(text)


class TokenizeBatch(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer, max_sequence_length: int = None):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def sample_windows(self, sequence, num_intervals):
        import random
        random.seed(42)
        """Phil's implement"""
        # Returns: a list of sampled windows.
        sampled_windows = []
        max_start_index = len(sequence) - self.max_sequence_length
        if max_start_index < 0:
            raise ValueError("max_sequence_length greater than length of sequence")
        for _ in range(num_intervals):
            # Randomly select start index of the window
            start_index = random.randint(0, max_start_index)
            # Extract window
            window = sequence[start_index:start_index + self.max_sequence_length]
            # Append the window to the list of sampled windows
            sampled_windows.append(window)
        return sampled_windows

    def __call__(self, batch: Dataset):
        """will complete tokenization, padding, and add eos"""
        proteins = [sample['sequences'] for sample in batch]
        data_tokens = self.tokenizer.encode(proteins)

        data_tokens_sampled = []
        for token_list in data_tokens:
            length = len(token_list)
            if length >= self.max_sequence_length:
                sampled_windows = self.sample_windows(token_list, length // self.max_sequence_length)
                for sample in sampled_windows:
                    data_tokens_sampled.append(sample)
            else:
                data_tokens_sampled.append(token_list)

        max_length_in_batch = max(len(t) for t in data_tokens_sampled)
        padded_tokenized_texts = [t + [0] * (max_length_in_batch - len(t)) for t in data_tokens_sampled]

        input_ids = torch.tensor(padded_tokenized_texts, dtype=torch.long)
        attention_mask = (input_ids != 0).int()

        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }
        return data


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
        self.vocab_size = '8k'
        self.tokenize_batch = ProteinTokenizer(self.vocab_size).get_batch_converter(self.max_sequence_length)

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


if __name__ == '__main__':
    dm = pretrainDataset(batch_size=1)
    # make sure dataset has "training" key
    print(type(dm.dataset['train']))
    print(len(dm.dataset['train']))
    print(len(dm.dataset['train']))