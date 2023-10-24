import pandas as pd
import sys
import glob
import pytorch_lightning as pl
import torch
import pickle
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from functools import partial
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime
from pytorch_lightning import seed_everything
import random
import gc
import h5py
import numpy as np

seed_everything(42)

global_tokenizer = None


def init_pool(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer


def standalone_tokenize_function(tup, extra_toks_per_seq=2):
    global global_tokenizer
    s1, s2 = tup
    try:
        tokens_1 = global_tokenizer.encode(s1)
        tokens_2 = global_tokenizer.encode(s2)
        return tokens_1, tokens_2
    except Exception as e:
        raise ValueError(f"Error during tokenization of string {s1} {s2}: {e}")


class TokenizeBatch(object):
    """add padding, create labels for GPT-alike training, used as collate_fn, need processed batch indices
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.unk_id()

    def __call__(self, batches):
        data_tokens = [torch.tensor(token_list) for token_list in batches]
        data_tokens_padded = pad_sequence(data_tokens, batch_first=True, padding_value=self.pad_token_id)

        # Create attention masks
        attention_masks = (data_tokens_padded != self.pad_token_id).long()

        # skip label==-100 during training so that these tokens won't be used in loss calculation
        labels = data_tokens_padded.clone()
        labels[data_tokens_padded == self.pad_token_id] = -100

        return {
            'input_ids': data_tokens_padded,
            'attention_mask': attention_masks,
            'labels': labels
        }


class BatchedPPIDataset(object):
    """inspired by esm2, but instead of sorting the original sequences,
    we should really sorting based on tokenized sequences
    """

    def __init__(self, sequence_strs, tokenizer, max_sequence_length):
        self.batch_indices = None
        self.sequence_str_1 = sequence_strs['sequence_1']
        self.sequence_str_2 = sequence_strs['sequence_2']
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.tokenized_sequences = []
        self.accumulated_length = 0
        # automatically tokenize sequences upon creation of the object
        # if need manual, change it to call object.process_all() in a separate line
        # self.tokenize_sequences()

    def tokenize_sequences_forward(self):
        prot_tuples = list(zip(self.sequence_str_1, self.sequence_str_2))

        with Pool(processes=8, initializer=init_pool, initargs=(self.tokenizer,)) as pool:
            tokenized_pairs = list(
                tqdm(pool.imap(partial(standalone_tokenize_function),
                               prot_tuples),
                     total=len(prot_tuples)))

        for tokens_1, tokens_2 in tokenized_pairs:
            seq_length = len(tokens_1) + len(tokens_2) + 2  # for both bos, eos tokens
            if seq_length <= self.max_sequence_length:
                forward_sequence = [self.tokenizer.bos_id()] + tokens_1 + [self.tokenizer.eos_id()] + tokens_2
                self.tokenized_sequences.append(forward_sequence)

    def tokenize_sequences_backward(self):
        """avoid re-tokenization of the same sequences"""
        self.reversed_tokenized_sequences = []
        for sequence in self.tokenized_sequences:
            eos_position = sequence.index(self.tokenizer.eos_id())
            tokens_1 = sequence[1:eos_position]  # Extract tokens1 without bos and eos
            tokens_2 = sequence[eos_position + 1:]  # Extract tokens2 after eos
            reversed_sequence = [self.tokenizer.bos_id()] + tokens_2 + [self.tokenizer.eos_id()] + tokens_1
            self.reversed_tokenized_sequences.append(reversed_sequence)
        self.tokenized_sequences.extend(self.reversed_tokenized_sequences)

    def process_all(self, base_path, split_name):
        self.tokenize_sequences_forward()
        forward_batches = self.process_chunk(self.tokenized_sequences, self.get_batch_indices())
        offset = len(self.tokenized_sequences)
        self.tokenize_sequences_backward()
        backward_batches = self.process_chunk(self.tokenized_sequences, self.get_batch_indices(offset))
        self.tokenized_sequences = []
        combined_dataset = concatenate_datasets([forward_batches, backward_batches])
        # shuffle the datasets overall again
        self.shuffled_dataset = combined_dataset.shuffle()
        self.save_checkpoint(self.shuffled_dataset, base_path,
                             split_name=split_name)
        return self.shuffled_dataset

    def process_chunk(self, tokenized_sequences, batch_indices):
        print(f'Start padding and masking for sequences {len(batch_indices)} batches')

        token_batch_fn = TokenizeBatch(self.tokenizer)
        processed_batches = [
            token_batch_fn([tokenized_sequences[i] for i in batch]) for batch
            in batch_indices]
        assert len(processed_batches) == len(batch_indices)

        # Shuffle together using a permutation
        permutation = list(torch.randperm(len(processed_batches)))
        processed_batches = [processed_batches[i] for i in permutation]

        all_attention_masks = []
        all_input_ids = []
        all_labels = []

        all_attention_masks.extend([batch['attention_mask'] for batch in processed_batches])
        all_input_ids.extend([batch['input_ids'] for batch in processed_batches])
        all_labels.extend([batch['labels'] for batch in processed_batches])

        combined_dataset = Dataset.from_dict({
            'attention_mask': all_attention_masks,
            'input_ids': all_input_ids,
            'labels': all_labels
        })
        del token_batch_fn, processed_batches, batch_indices, tokenized_sequences, all_attention_masks, all_input_ids, all_labels
        gc.collect()

        return combined_dataset

    def save_checkpoint(self, shuffled_dataset, base_path, split_name=None):
        print(f'Start generating tokens for shuffled_dataset sequences')
        # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_file = f'{base_path}/{split_name}_combined_reversed_ppi_tokenized_sequences.hf'

        # clear up memory for the multiprocessing
        shuffled_dataset.save_to_disk(output_file)
        del shuffled_dataset
        print(f'successfully written {split_name} processed datasets into disc!')
        self.tokenized_sequences.clear()
        gc.collect()

    def get_batch_indices(self, offset=0, end=None):
        if end is None:
            end = len(self.tokenized_sequences)
        # list splice to isolate processing of forward and backward sequences who
        # are both stored within self.tokenized_sequences
        sizes = [(len(tokens), i) for i, tokens in enumerate(self.tokenized_sequences[offset:end])]
        sizes = [(sz, idx + offset) for sz, idx in sizes]
        sizes.sort()
        batches = []
        buf = []
        current_buf_len = 0

        def _flush_current_buf():
            nonlocal current_buf_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            current_buf_len = 0
            # print('my batches is:')
            # print(batches)

        for sz, i in sizes:
            # sz already has the length of special tokens, handled at tokenization level
            # check accumulative seq length in the buffer
            if current_buf_len + sz > self.max_sequence_length:
                _flush_current_buf()
            buf.append(i)
            current_buf_len += sz
            # print('my buffer is:')
            # print(buf)

        _flush_current_buf()
        return batches

    def get_sequence_lengths(self):
        return [len(seq) for batch in self.shuffled_dataset['input_ids'] for seq in batch]

import matplotlib.pyplot as plt

def plot_histogram(lengths, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__=='__main__':
    path = '/data/rozen/home/e0833634/lama/data/ppi_8000'
    train_dataset = load_from_disk(path + '/ppi_8000_raw.hf')
    import sentencepiece as spm
    vocab_size = '8k'
    tokenizer_path = '/data/rozen/home/e0833634/lama/protllama/batch_script/'
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path + f"protein_{vocab_size}.model")
    max_length = 1024
    batched_dataset = BatchedPPIDataset(train_dataset['train'], tokenizer, max_length)
    processed_dataset = batched_dataset.process_all('/data/rozen/home/e0833634/lama/protllama/check_token_length', 'train')
    print(processed_dataset)
    sequence_lengths = batched_dataset.get_sequence_lengths()
    valid_sequences_count = len([length for length in sequence_lengths if length <= max_length])
    percentage = valid_sequences_count / 862276 * 100
    print(f"Total number of sequences with token length <= {max_length}: {valid_sequences_count}, {percentage:.3f}%")
    plot_histogram(sequence_lengths, f"Token Length Distribution of Training ({vocab_size} vocab)", "Token Length",
                   "Number of Sequences", f'/data/rozen/home/e0833634/lama/protllama/check_token_length/distribution_{vocab_size}_{max_length}.png')

