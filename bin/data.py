import sys
import glob
import pytorch_lightning as pl
import torch
import pickle
from datasets import Dataset, DatasetDict, load_from_disk
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


def standalone_tokenize_function(s, max_sequence_length):
    global global_tokenizer
    try:
        tokens = global_tokenizer.encode(s)
        tokenized_sequence = []
        if len(tokens) > max_sequence_length - 1:
            sampled_windows = sample_windows(tokens, max_sequence_length)
            for sample in sampled_windows:
                sample.insert(0, global_tokenizer.bos_id())
                tokenized_sequence.append(sample)
        else:
            tokens.insert(0, global_tokenizer.bos_id())
            tokenized_sequence.append(tokens)
        return tokenized_sequence
    except Exception as e:
        raise ValueError(f"Error during tokenization of string {s}: {e}")


def sample_windows(sequence, max_sequence_length, extra_toks_per_seq=1):
    sampled_windows = []
    sampled_windows.append(sequence[:max_sequence_length - extra_toks_per_seq])
    num_slices_required = (len(sequence) // max_sequence_length) - 2
    max_start_index = len(sequence) - max_sequence_length

    if max_start_index < 0:
        raise ValueError("max_sequence_length greater than length of sequence")

    if num_slices_required > 0:
        for _ in range(num_slices_required):
            start_index = random.randint(0, max_start_index)
            window = sequence[start_index:start_index + max_sequence_length - extra_toks_per_seq]
            sampled_windows.append(window)

    sampled_windows.append(sequence[-(max_sequence_length - extra_toks_per_seq):])
    return sampled_windows


class TokenizeBatch(object):
    """add padding, create labels for GPT-alike training, used as collate_fn, need processed batch indices
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_id()

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


class BatchedDataset(object):
    """inspired by esm2, but instead of sorting the original sequences,
    we should really sorting based on tokenized sequences
    """

    def __init__(self, sequence_strs, tokenizer, max_sequence_length):
        self.batch_indices = None
        self.sequence_strs = sequence_strs['sequences']
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.tokenized_sequences = []
        self.accumulated_length = 0
        # automatically tokenize sequences upon creation of the object
        # if need manual, change it to call object.tokenize_sequence() in a separate line
        # self.tokenize_sequences()

    def tokenize_sequences(self, base_path, split_name):
        checkpoint_interval = 1000000

        # any checkpoints?
        start_idx, file_name = self.get_latest_checkpoint_sequence(base_path, split_name)

        if start_idx > 0:
            with open(file_name, 'rb') as f:
                _, self.accumulated_length = pickle.load(f)
                # We initialize tokenized_sequences as an empty list because previous tokenized data has
                # already been processed and saved in previous runs.
                self.tokenized_sequences = []
                print(f'Resume training from {start_idx}')
                print(f'Actual tokenized sequence number is {self.accumulated_length}')

        with Pool(processes=16, initializer=init_pool, initargs=(self.tokenizer,)) as pool:
            for idx, result in enumerate(
                    tqdm(pool.imap(partial(standalone_tokenize_function, max_sequence_length=self.max_sequence_length),
                                   self.sequence_strs[start_idx:]),
                         total=len(self.sequence_strs) - start_idx)):
                self.tokenized_sequences.extend(result)
                if ((idx + start_idx) % checkpoint_interval == 0) & (idx != 0):
                    batch_indices = self.get_batch_indices(offset=self.accumulated_length)
                    self.accumulated_length += len(self.tokenized_sequences)
                    self.save_checkpoint(idx + start_idx, base_path, batch_indices=batch_indices, split_name=split_name)
            # save the last model
            batch_indices = self.get_batch_indices(offset=self.accumulated_length)
            self.accumulated_length += len(self.tokenized_sequences)
            self.save_checkpoint(len(self.sequence_strs), base_path, batch_indices=batch_indices, split_name=split_name)

    def process_chunk(self, base_path, tokenized_sequences, batch_indices, idx, split_name):
        token_batch_fn = TokenizeBatch(self.tokenizer)
        processed_batches = [
            token_batch_fn([tokenized_sequences[i] for i in [idx - self.accumulated_length for idx in batch]]) for batch
            in batch_indices]
        assert len(processed_batches) == len(batch_indices)

        # Shuffle together using a permutation
        permutation = list(torch.randperm(len(processed_batches)))
        processed_batches = [processed_batches[i] for i in permutation]
        batch_indices = [batch_indices[i] for i in permutation]

        processed_filename = f"{base_path}/{split_name}_intermediate_processed_dataset_{idx}.pkl"
        with open(processed_filename, 'wb') as f:
            pickle.dump(processed_batches, f)
        print(f'Finish padding and masking for {idx} sequences')
        with open(f'{base_path}/{split_name}_intermediate_checkpoint_batches_{idx}.pkl', 'wb') as f:
            pickle.dump(batch_indices, f)

        del token_batch_fn, processed_batches, batch_indices, tokenized_sequences
        gc.collect()

    def save_checkpoint(self, idx, base_path, batch_indices=None, split_name=None):
        print(f'Start generating tokens for {idx} sequences')
        # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with open(f'{base_path}/{split_name}_intermediate_checkpoint_{idx}.pkl', 'wb') as f:
            pickle.dump((self.tokenized_sequences, self.accumulated_length), f)

        print(f'Start padding and masking for {idx} sequences {len(batch_indices)} batches')
        self.process_chunk(base_path, self.tokenized_sequences, batch_indices, idx, split_name)

        # clear up memory for the multiprocessing
        self.tokenized_sequences.clear()
        gc.collect()
        print(f'Finish generating tokens for {idx} strs')

    @staticmethod
    def get_latest_checkpoint_sequence(base_path, split_name):
        ckpt_path = f'{base_path}/{split_name}_intermediate_checkpoint_*.pkl'
        cache_list = glob.glob(ckpt_path)
        if cache_list:
            cache_fname = max(cache_list, key=os.path.getctime)
            # since we might need the processed dataset, revert the system to redo this checkpoint
            index = int(cache_fname.split('_')[4].split('.')[0]) - 1000000
            # tell the model: original sequences up to this index is processed already, start from 0 for subsets again
            fname = f'{base_path}/{split_name}_intermediate_checkpoint_{index}.pkl'
            return index, fname
        else:
            return 0, None

    @staticmethod
    def combine_checkpoints(base_path, split_name, batch_path):
        dataset_ckpt_path = f'{base_path}/{split_name}_intermediate_processed_dataset_*.pkl'
        batch_idx_ckpt_path = f'{base_path}/{split_name}_intermediate_checkpoint_batches_*.pkl'
        output_file = f'{base_path}/{split_name}_combined_tokenized_sequences.hf'

        all_attention_masks = []
        all_input_ids = []
        all_labels = []

        for dataset_file in sorted(glob.glob(dataset_ckpt_path), key=lambda x: int(x.split('_')[-1].split('.')[0])):
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
                all_attention_masks.extend([batch['attention_mask'] for batch in dataset])
                all_input_ids.extend([batch['input_ids'] for batch in dataset])
                all_labels.extend([batch['labels'] for batch in dataset])

        combined_dataset = Dataset.from_dict({
            'attention_mask': all_attention_masks,
            'input_ids': all_input_ids,
            'labels': all_labels
        })

        combined_dataset.save_to_disk(output_file)
        del combined_dataset, all_attention_masks, all_input_ids, all_labels
        print('successfully written all processed datasets into disc!')

        print('start to combine indices...')
        all_batch_indices = []
        for batch_file in sorted(glob.glob(batch_idx_ckpt_path), key=lambda x: int(x.split('_')[-1].split('.')[0])):
            with open(batch_file, 'rb') as f:
                batch_indices = pickle.load(f)
                all_batch_indices.extend(batch_indices)

        # Save the combined batch indices
        with open(f"{batch_path}_{split_name}_Batch_indices.pkl",
                  "wb") as f:
            pickle.dump(all_batch_indices, f)

    def get_batch_indices(self, extra_toks_per_seq=1, offset=0):
        sizes = [(len(tokens), i) for i, tokens in enumerate(self.tokenized_sequences)]
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
            # considering the extra bos
            sz += extra_toks_per_seq
            # check accumulative seq length in the buffer
            if current_buf_len + sz > self.max_sequence_length:
                _flush_current_buf()
            buf.append(i)
            current_buf_len += sz
            # print('my buffer is:')
            # print(buf)

        _flush_current_buf()
        return batches


class DynamicBatchingDataset(Dataset):
    """
    Process dynamically batched datasets of Huggingface Datasets object. Need special handling since in the previous
    steps, each batch (row in the Datasets object) is already processed for per batch loading
    Usage:
    train_dataset = DynamicBatchingDataset(small_dataset_dict['train'], batch_indices_train)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False,
                        collate_fn=DynamicBatchingDataset.dynamic_padding_collate_fn)
    """

    def __init__(self, dataset_dict, batch_indices):
        print('Initializing dataset...')
        self.dataset_dict = dataset_dict
        self.batch_indices = batch_indices  # This is mainly for informational purposes, if needed.

    def __len__(self):
        return len(self.dataset_dict['attention_mask'])  # assuming each entry in dataset_dict represents a batch

    def __getitem__(self, idx):
        # Directly retrieve the batch using the index
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
    def dynamic_padding_collate_fn(batch):
        """
        Args:
            batch: preprocessed datasets, already go through dynamic batching, so each batch is upto ~maximum_seq_length,
            eg. 512 tokens length.
            That resulted in each batch input shapes being different, in both sequence numbers and sequence lengths,
            (although sequence lengths being constant within a batch).
            For DDP, when distribute dataset between GPUs, the shapes of input have to be the same, therefore, need
            extra paddings when loading datasets to the GPUs.

        Returns:
            Dynamically padded tokens, according to the maximum length of the batch across all the batches
            loaded to the GPUs per iteration.
            Slower but more efficient than just pad to the maximum limit.
        """

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


class PretrainDataset(pl.LightningDataModule):
    def __init__(self,
                 tokenizer_path,
                 input_dataset_path,
                 output_dataset_path,
                 num_workers,
                 batch_size,
                 target: str = 'protein',
                 max_sequence_length: int = 512,
                 vocab_size='8k',
                 ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.intermediate_data_path = str(f'{output_dataset_path}/intermediate')
        if not os.path.exists(self.intermediate_data_path):
            os.makedirs(self.intermediate_data_path)
        self.vocab_size = vocab_size
        self.num_workers = num_workers
        self.target = target
        self.original_data = self.retrieve_data(input_dataset_path, self.target)
        self.tokenizer = self.tokenizer_generation(tokenizer_path, self.target, self.vocab_size)
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size  # used for DDP, determines how many batches load simultaneously \
                                    # to multiple GPUs at a time. Not used for tokenization.
        self.dataset_path = f'{output_dataset_path}/uniref50_random90split_{self.vocab_size}_{self.max_sequence_length}_1million_dataset.hf'
        self.batch_path = f'{output_dataset_path}/uniref50_random90split_{self.vocab_size}_{self.max_sequence_length}'

        if not os.path.exists(self.dataset_path):
            print('Start generating tokenized datasets')
            self.tokenized_data = {}
            self.save_tokenized_data()
        else:
            print('Load processed datasets')
            self.dataset = load_from_disk(self.dataset_path)

    @staticmethod
    def retrieve_data(input_dataset_path, target):
        """
        input: transformers DATASET object, with train/valid split
        """
        if target == 'original':
            with open('/data/rozen/home/e0833634/lama/protllama/original_lama.pkl', 'rb') as f:
                loaded_data_lama = pickle.load(f)
            return loaded_data_lama
        elif target == 'protein':
            return load_from_disk(f'{input_dataset_path}/uniref50_random90split.hf')
        else:
            raise ValueError('Have not prepared dataset for this target')

    @staticmethod
    def tokenizer_generation(tokenizer_path, target, vocab_size):
        if target == 'original':
            tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
            tokenizer.pad_token = tokenizer.unk_token
            return tokenizer
        elif target == 'protein':
            tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path + "protein_%s.model" % (vocab_size))
            return tokenizer
        else:
            raise ValueError('Have not prepared tokenizer for this target')

    def process_and_store_data(self, split_name):
        assert split_name in ["train", "valid"], "Invalid split_name. It should be either 'train' or 'valid'."

        print("Tokenizing sequences...")
        batched_dataset = BatchedDataset(self.original_data[split_name], self.tokenizer, self.max_sequence_length)
        batched_dataset.tokenize_sequences(base_path=self.intermediate_data_path, split_name=split_name)
        batched_dataset.combine_checkpoints(base_path=self.intermediate_data_path, split_name=split_name,
                                            batch_path=self.batch_path)

        del batched_dataset
        gc.collect()

        print('Finish generating dataloader for ', split_name)

    # def shuffle_dataset(self, dataset):
    # shuffled_dataset = dataset.shuffle()
    # return shuffled_dataset

    def save_tokenized_data(self):
        for split_name in ['train', 'valid']:
            self.process_and_store_data(split_name)
        del self.original_data

        train_combined_path = f'{self.intermediate_data_path}/train_combined_tokenized_sequences.hf'
        validation_combined_path = f'{self.intermediate_data_path}/valid_combined_tokenized_sequences.hf'

        # Load them
        train_dataset = load_from_disk(train_combined_path)
        validation_dataset = load_from_disk(validation_combined_path)

        print('Start combining datasets')

        combined_datasets = DatasetDict({
            'train': train_dataset,
            'valid': validation_dataset
        })

        # If you want to save the combined dataset:
        combined_datasets.save_to_disk(self.dataset_path)
        del combined_datasets
        gc.collect()
        # with open(self.dataset_path, "wb") as file:
        # pickle.dump(self.dataset, file)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        with open(f"{self.batch_path}_train_Batch_indices.pkl", 'rb') as f:
            batch_indices_train = pickle.load(f)
        with open(f"{self.batch_path}_valid_Batch_indices.pkl", 'rb') as f:
            batch_indices_val = pickle.load(f)
        self.train_dataset = DynamicBatchingDataset(self.dataset['train'], batch_indices_train)
        # Repeat similar steps for validation and test datasets if needed
        self.val_dataset = DynamicBatchingDataset(self.dataset['valid'], batch_indices_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=DynamicBatchingDataset.dynamic_padding_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=DynamicBatchingDataset.dynamic_padding_collate_fn)
