import sys
import glob
import lightning.pytorch as pl
import torch
import pickle
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from functools import partial
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime
import random
import gc
import h5py
import numpy as np
from multiprocessing import Pool

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

        with Pool(processes=16, initializer=init_pool, initargs=(self.tokenizer,)) as pool:
            tokenized_pairs = list(
                tqdm(pool.imap(partial(standalone_tokenize_function),
                               prot_tuples),
                     total=len(prot_tuples)))

        for tokens_1, tokens_2 in tokenized_pairs:
            seq_length = len(tokens_1) + len(tokens_2) + 3  # for both bos, eos, sep tokens
            if seq_length <= self.max_sequence_length:
                forward_sequence = [self.tokenizer.bos_id()] + tokens_1 + [self.tokenizer.piece_to_id('<sep>')] + tokens_2 + [self.tokenizer.eos_id()]
                self.tokenized_sequences.append(forward_sequence)

    def tokenize_sequences_backward(self):
        """avoid re-tokenization of the same sequences"""
        self.reversed_tokenized_sequences = []
        for sequence in self.tokenized_sequences:
            sep_position = sequence.index(self.tokenizer.piece_to_id('<sep>'))
            tokens_1 = sequence[1:sep_position]  # Extract tokens1 without bos and sep
            tokens_2 = sequence[sep_position + 1:-1]  # Extract tokens2 after sep before eos
            reversed_sequence = [self.tokenizer.bos_id()] + tokens_2 + [self.tokenizer.piece_to_id('<sep>')] + tokens_1 + [self.tokenizer.eos_id()]
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
        shuffled_dataset = combined_dataset.shuffle()
        self.save_checkpoint(shuffled_dataset, base_path,
                             split_name=split_name)
        return shuffled_dataset

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


class DynamicBatchingDataset(Dataset):
    """
    Process dynamically batched datasets of Huggingface Datasets object. Need special handling since in the previous
    steps, each batch (row in the Datasets object) is already processed for per batch loading
    Usage:
    train_dataset = DynamicBatchingDataset(small_dataset_dict['train'], batch_indices_train)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False,
                        collate_fn=DynamicBatchingDataset.dynamic_padding_collate_fn)
    """

    def __init__(self, dataset_dict):
        print('Initializing dataset...')
        #self.dataset_dict = dataset_dict
        self.dataset_dict = {
            'attention_mask': [torch.tensor(item) for item in dataset_dict['attention_mask']],
            'input_ids': [torch.tensor(item) for item in dataset_dict['input_ids']],
            'labels': [torch.tensor(item) for item in dataset_dict['labels']]
        }

    def __len__(self):
        return len(self.dataset_dict['attention_mask'])  # assuming each entry in dataset_dict represents a batch

    def __getitem__(self, idx):
        # Check if idx is an integer or a list
        if isinstance(idx, int):
            return {
                'attention_mask': self.dataset_dict['attention_mask'][idx],
                'input_ids': self.dataset_dict['input_ids'][idx],
                'labels': self.dataset_dict['labels'][idx]
            }
        elif isinstance(idx, list):
            return {
                'attention_mask': [self.dataset_dict['attention_mask'][i] for i in idx],
                'input_ids': [self.dataset_dict['input_ids'][i] for i in idx],
                'labels': [self.dataset_dict['labels'][i] for i in idx]
            }   
        else:
            raise ValueError(f"Expected idx to be int or list, but got {type(idx)}")    
        
    #if isinstance(idx, int):
         #   indices = [idx]
        #else:
        #    indices = idx
        
        #attention_masks = []
        #input_ids = []
        #labels = []
        #for index in indices:
         #   attention_masks.append(torch.tensor(self.dataset_dict['attention_mask'][index]))
         #   input_ids.append(torch.tensor(self.dataset_dict['input_ids'][index]))
         #   labels.append(torch.tensor(self.dataset_dict['labels'][index]))

        #return {
         #   'attention_mask': attention_masks,
          #  'input_ids': input_ids,
          #  'labels': labels
        #}

    @staticmethod
    def collate_fn(batch, verbose=False):
        # Since DataLoader's batch_size is 1, batch[0] contains your pre-batched data
        item = batch[0]
        #if verbose:
         #   print(f"collate_fn batch shape: {item['input_ids'].shape}")

       # attention_mask = item['attention_mask']
       # input_ids = item['input_ids']
       # if verbose:
        #    print(f"collate_fn input_ids shape after indexing: {input_ids.shape}")
        #labels = item['labels']

        # These are already pre-padded, so you can directly return
        return {
            'attention_mask': item['attention_mask'],
            'input_ids': item['input_ids'],
            'labels': item['labels']
        }

class PretrainPPIDataset(pl.LightningDataModule):
    def __init__(self,
                 tokenizer_path,
                 input_dataset_path,
                 output_dataset_path,
                 num_workers,
                 batch_size,
                 testornot: bool = False,
                 target: str = 'protein',
                 max_sequence_length: int = 512,
                 vocab_size='8k',
                 ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.need_test = testornot
        self.test_dataset = None
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
        self.dataset_path = f'{output_dataset_path}/ppi_8000_{self.vocab_size}_{self.max_sequence_length}_dataset_special.hf'

    def prepare_data(self):
        if not os.path.exists(self.dataset_path):
            print('Start generating tokenized datasets')
            self.tokenized_data = {}
            self.save_tokenized_data()
            #self.dataset = load_from_disk(self.dataset_path)
            #small test set if needed
            #self.dataset = DatasetDict({
               # 'train': dataset['train'].select(range(10)),
              # 'valid': dataset['valid'].select(range(10))})
    
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
        elif target == 'ppi':
            #return load_from_disk(f'{input_dataset_path}/ppi_8000_raw.hf')
            return load_from_disk(f'{input_dataset_path}/ppi_noisy_raw.hf')
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
        elif target == 'ppi':
            tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path + "protein_%s_special.model" % (vocab_size))
            return tokenizer
        else:
            raise ValueError('Have not prepared tokenizer for this target')

    def process_and_store_data(self, split_name):
        assert split_name in ["train", "valid", "test"], "Invalid split_name. It should be either 'train' or 'valid' or 'test'."

        print("Tokenizing sequences...")
        batched_dataset = BatchedPPIDataset(self.original_data[split_name], self.tokenizer, self.max_sequence_length)
        batched_dataset.process_all(base_path=self.intermediate_data_path, split_name=split_name)

        del batched_dataset
        gc.collect()

        print('Finish generating dataloader for ', split_name)

    def save_tokenized_data(self):
        if self.need_test:
            for split_name in ['train', 'valid', 'test']:
                self.process_and_store_data(split_name)
        else:
            for split_name in ['train', 'valid']:
                self.process_and_store_data(split_name)
        del self.original_data

        train_combined_path = f'{self.intermediate_data_path}/train_combined_reversed_ppi_tokenized_sequences.hf'
        validation_combined_path = f'{self.intermediate_data_path}/valid_combined_reversed_ppi_tokenized_sequences.hf'
        if self.need_test:
            test_combined_path = f'{self.intermediate_data_path}/test_combined_reversed_ppi_tokenized_sequences.hf'

        # Load them
        train_dataset = load_from_disk(train_combined_path)
        validation_dataset = load_from_disk(validation_combined_path)
        if self.need_test:
            test_dataset = load_from_disk(test_combined_path)
            combined_datasets = DatasetDict({
                'train': train_dataset,
                'valid': validation_dataset,
                'test': test_dataset
            })

            print('Start combining datasets')
        else:
            combined_datasets = DatasetDict({
                'train': train_dataset,
                'valid': validation_dataset
            })

        # If you want to save the combined dataset:
        combined_datasets.save_to_disk(self.dataset_path)
        del combined_datasets, train_dataset, validation_dataset
        gc.collect()
        # with open(self.dataset_path, "wb") as file:
        # pickle.dump(self.dataset, file)
    
    @staticmethod
    def shard_dataset_for_gpu(args):
        rank, num_gpus, dataset, dataset_length = args

        shard_length = dataset_length // num_gpus
        start_idx = shard_length * rank
        end_idx = start_idx + shard_length if rank != num_gpus - 1 else dataset_length

        subset_data = DictSubset(dataset['train'], list(range(start_idx, end_idx)))
        processed_subset = {
            'attention_mask': [subset_data[i]['attention_mask'] for i in range(len(subset_data))],
            'input_ids': [subset_data[i]['input_ids'] for i in range(len(subset_data))],
            'labels': [subset_data[i]['labels'] for i in range(len(subset_data))]
        }
        subset_dataset = DynamicBatchingDataset(processed_subset)
        return rank, subset_dataset


    def setup(self, stage: str):
        #global_rank = self.trainer.global_rank
        #shard_path = f'/home/a03-yzhang/projects/protllama2_output/ppi_8000/ppi_8000_10k_2048_dataset.hf/train/data-0000{global_rank}-of-00009.arrow'
        #self.shard = Dataset.from_file(shard_path)
        #valid_path = self.dataset_path+'/valid'
        self.dataset = load_from_disk(self.dataset_path)
        #self.valid_dataset = load_from_disk(valid_path)
        #self.train_dataset = DynamicBatchingDataset(self.shard)
        self.train_dataset = DynamicBatchingDataset(self.dataset['train'])
        self.val_dataset = DynamicBatchingDataset(self.dataset['valid'])
        if self.need_test:
            self.test_dataset = DynamicBatchingDataset(self.dataset['test'])
        #self.val_dataset = DynamicBatchingDataset(self.valid_dataset)

        gc.collect()

    def train_dataloader(self):
        print('Building training dataloader')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=DynamicBatchingDataset.collate_fn, pin_memory=True)

    def val_dataloader(self):
        print('Building validation dataloader')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=DynamicBatchingDataset.collate_fn, pin_memory=True)

    def test_dataloader(self):
        if self.need_test:
            print('Building test dataloader')
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=DynamicBatchingDataset.collate_fn, pin_memory=True)
        else:
            pass

if __name__=='__main__':
    dm = PretrainPPIDataset(
        input_dataset_path='/data/rozen/home/e0833634/lama/data/ppi_8000',
        output_dataset_path='/data/rozen/home/e0833634/lama/data/ppi_8000',
        tokenizer_path='/data/rozen/home/e0833634/lama/protllama/batch_script/',
        num_workers=2,
        batch_size=1,
        vocab_size='32k',
        target='ppi',
        testornot=False,
        max_sequence_length=2048,
    )
    dm.prepare_data()
    dm.setup('fit')
    dm.train_dataloader()
    dm.val_dataloader()
