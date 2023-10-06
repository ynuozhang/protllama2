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
        # raw_tokenized_sequences = [self.tokenizer.encode(s) for s in self.sequence_strs]

        self.tokenized_sequences = []
        self.accumulated_length = 0
        # automatically tokenize sequences upon creation of the object
        # if need manual, change it to call object.tokenize_sequence() in a separate line
        # self.tokenize_sequences()

    def tokenize_sequences(self, split_name):
        checkpoint_interval = 1000000

        # any checkpoints?
        start_idx, file_name = self.get_latest_checkpoint_sequence(split_name)

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
                    self.save_checkpoint(idx + start_idx, batch_indices=batch_indices, split_name=split_name)
            # save the last model
            batch_indices = self.get_batch_indices(offset=self.accumulated_length)
            self.accumulated_length += len(self.tokenized_sequences)
            #batch_indices = self.get_batch_indices()
            self.save_checkpoint(len(self.sequence_strs), batch_indices=batch_indices, split_name=split_name)

    def process_chunk(self, tokenized_sequences, batch_indices, idx, split_name):
        token_batch_fn = TokenizeBatch(self.tokenizer)
        #processed_batches = [token_batch_fn([tokenized_sequences[i] for i in batch]) for batch in (batch_indices-self.accumulated_length)]
        processed_batches = [
            token_batch_fn([tokenized_sequences[i] for i in [idx - self.accumulated_length for idx in batch]]) for batch
            in batch_indices]
        assert len(processed_batches) == len(batch_indices)

        # Shuffle together using a permutation
        permutation = list(torch.randperm(len(processed_batches)))
        processed_batches = [processed_batches[i] for i in permutation]
        batch_indices = [batch_indices[i] for i in permutation]

        processed_filename = f"{split_name}_intermediate_processed_dataset_{idx}.pkl"
        with open(processed_filename, 'wb') as f:
            pickle.dump(processed_batches, f)
        print(f'Finish padding and masking for {idx} sequences')
        with open(f'{split_name}_intermediate_checkpoint_batches_{idx}.pkl', 'wb') as f:
            pickle.dump(batch_indices, f)

        del token_batch_fn, processed_batches, batch_indices, tokenized_sequences
        gc.collect()

    def save_checkpoint(self, idx, batch_indices=None, split_name=None):
        print(f'Start generating tokens for {idx} sequences')
        #current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with open(f'{split_name}_intermediate_checkpoint_{idx}.pkl', 'wb') as f:
            pickle.dump((self.tokenized_sequences, self.accumulated_length), f)

        print(f'Start padding and masking for {idx} sequences {len(batch_indices)} batches')
        self.process_chunk(self.tokenized_sequences, batch_indices, idx, split_name)

        # clear up memory for the multiprocessing
        self.tokenized_sequences.clear()
        gc.collect()
        print(f'Finish generating tokens for {idx} strs')

    @staticmethod
    def get_latest_checkpoint_sequence(split_name):
        ckpt_path = f'/data/rozen/home/e0833634/lama/protllama/batch_script/{split_name}_intermediate_checkpoint_*.pkl'
        cache_list = glob.glob(ckpt_path)
        if cache_list:
            cache_fname = max(cache_list, key=os.path.getctime)
            # since we might need the processed dataset, revert the system to redo this checkpoint
            index = int(cache_fname.split('_')[4].split('.')[0]) - 1000000
            # tell the model: original sequences up to this index is processed already, start from 0 for subsets again
            fname = f'/data/rozen/home/e0833634/lama/protllama/batch_script/{split_name}_intermediate_checkpoint_{index}.pkl'
            return index, fname
        else:
            return 0, None

    @staticmethod
    def combine_checkpoints(split_name, batch_path):
        dataset_ckpt_path = f'/data/rozen/home/e0833634/lama/protllama/batch_script/{split_name}_intermediate_processed_dataset_*.pkl'
        batch_idx_ckpt_path = f'/data/rozen/home/e0833634/lama/protllama/batch_script/{split_name}_intermediate_checkpoint_batches_*.pkl'
        output_file = f'/data/rozen/home/e0833634/lama/protllama/batch_script/{split_name}_combined_tokenized_sequences.hf'

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

        # Save the combined batch indices (if needed)
        with open(f"{batch_path}_{split_name}_Batch_indices.pkl",
                  "wb") as f:
            pickle.dump(all_batch_indices, f)

    def load_combined_dataset(self, split_name, combined_file):
        self.processed_sequences = load_from_disk(combined_file)
        print('Load combined processed sequences from ', combined_file)
        #with open(f"/data/rozen/home/e0833634/lama/protllama/batch_script/{split_name}_Batch_indices.pkl", 'rb') as f:
            #self.batch_indices = pickle.load(f)
        #print('Loaded batch indices')

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

    def __len__(self):
        return len(self.processed_sequences)

    def __getitem__(self, idx):
        return self.processed_sequences[idx]


class PretrainDataset(pl.LightningDataModule):
    def __init__(self,
                 # batch_size: int = 1,
                 target: str = 'protein',
                 max_sequence_length: int = 512):
        super().__init__()

        self.vocab_size = '8k'
        self.target = target
        self.original_data = self.retrieve_data(self.target)
        self.tokenizer = self.tokenizer_generation(self.target, self.vocab_size)
        self.max_sequence_length = max_sequence_length
        self.dataset_path = f'/data/rozen/home/e0833634/lama/data/swiss_2023_9/uniref50_random90split_{self.vocab_size}_{self.max_sequence_length}_Dataset.hf'
        self.batch_path = f'/data/rozen/home/e0833634/lama/data/swiss_2023_9/uniref50_random90split_{self.vocab_size}_{self.max_sequence_length}'

        if not os.path.exists(self.dataset_path):
            print('Start generating tokenized datasets')
            self.tokenized_data = {}
            self.save_tokenized_data()
        else:
            print('Load processed datasets')
            # with open(self.dataset_path, 'rb') as f:
            # self.dataset = pickle.load(f)
            self.dataset = load_from_disk(self.dataset_path)
        # self.batch_size = batch_size

    @staticmethod
    def retrieve_data(target):
        """
        input: transformers DATASET object, with train/valid split
        """
        if target == 'original':
            with open('/data/rozen/home/e0833634/lama/protllama/original_lama.pkl', 'rb') as f:
                loaded_data_lama = pickle.load(f)
            return loaded_data_lama
        elif target == 'protein':
            # with open('/data/rozen/home/e0833634/lama/data/swiss_2023_9/uniref50_random90split.pkl', 'rb') as f:
            # loaded_data = pickle.load(f)
            # return loaded_data
            return load_from_disk('/data/rozen/home/e0833634/lama/data/swiss_2023_9/uniref50_random90split.hf')
        else:
            raise ValueError('Have not prepared dataset for this target')

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

    def process_and_store_data(self, split_name):
        assert split_name in ["train", "valid"], "Invalid split_name. It should be either 'train' or 'valid'."

        combined_tokenized_sequences_path = f'/data/rozen/home/e0833634/lama/protllama/batch_script/{split_name}_combined_tokenized_sequences.hf'

        if os.path.exists(combined_tokenized_sequences_path):
            #print("Loading from combined dataset...")
            #batched_dataset = BatchedDataset(self.original_data[split_name], self.tokenizer, self.max_sequence_length)
            #batched_dataset.load_combined_dataset(split_name, combined_tokenized_sequences_path)
            pass
        else:
            print("Tokenizing sequences...")
            batched_dataset = BatchedDataset(self.original_data[split_name], self.tokenizer, self.max_sequence_length)
            batched_dataset.tokenize_sequences(split_name=split_name)
            batched_dataset.combine_checkpoints(split_name=split_name, batch_path = self.batch_path)
            #batched_dataset.load_combined_dataset(split_name, combined_tokenized_sequences_path)

        del self.original_data, batched_dataset
        gc.collect()

        #batches_indices = batched_dataset.batch_indices # use batch indices from the loaded/processed dataset
        #print(f"Processing {len(batches_indices)} batches for {split_name} split...")

        #with open(self.batch_path + '_' + split_name + '_Batch_indices.pkl', "wb") as file:
            #pickle.dump(batches_indices, file)
        print('Finish generating dataloader for ', split_name)

    #def shuffle_dataset(self, dataset):
        #shuffled_dataset = dataset.shuffle()
        #return shuffled_dataset

    def save_tokenized_data(self):
        for split_name in ['train', 'valid']:
            self.process_and_store_data(split_name)

        train_combined_path = '/data/rozen/home/e0833634/lama/protllama/batch_script/train_combined_tokenized_sequences.hf'
        validation_combined_path = '/data/rozen/home/e0833634/lama/protllama/batch_script/valid_combined_tokenized_sequences.hf'

        # Load them
        train_dataset = load_from_disk(train_combined_path)
        validation_dataset = load_from_disk(validation_combined_path)

        # Concatenate the datasets
        combined_dataset = concatenate_datasets([train_dataset, validation_dataset])

        # If you want to save the combined dataset:
        combined_dataset.save_to_disk(self.dataset_path)
        gc.collect()
        # with open(self.dataset_path, "wb") as file:
        # pickle.dump(self.dataset, file)

    def dataloader_preprocessing(self, split_name):
        ds = self.dataset[split_name]
        with open(self.batch_path + '_' + split_name + '_Batch_indices.pkl', 'rb') as file:
            batches = pickle.load(file)
        return torch.utils.data.DataLoader(ds, batch_sampler=batches, pin_memory=True)

    def train_dataloader(self):
        # if 'train' in self.dataset.keys():
        # train_ds = self.dataset['train']
        # dataset = BatchedDataset(train_ds, self.tokenizer, self.max_sequence_length)
        # batches = dataset.get_batch_indices()
        # return torch.utils.data.DataLoader(dataset, collate_fn=TokenizeBatch(self.tokenizer),
        # batch_sampler=batches, pin_memory=True)
        # return DataLoader(train_ds, batch_size=self.batch_size, shuffle=False,
        # drop_last=True,
        # collate_fn=partial(self.tokenize_batch))
        # elif 'train' not in self.dataset.keys():
        # raise ValueError('Did not detect training dataset')
        return self.dataloader_preprocessing('train')

    def val_dataloader(self):
        # if 'valid' in self.dataset.keys():
        # val_ds = self.dataset['valid']
        # dataset = BatchedDataset(val_ds, self.tokenizer, self.max_sequence_length)
        # batches = dataset.get_batch_indices()
        # return torch.utils.data.DataLoader(dataset, collate_fn=TokenizeBatch(self.tokenizer),
        # batch_sampler=batches, pin_memory=True)
        # return DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
        # drop_last=False,
        # collate_fn=partial(self.tokenize_batch))
        # else:
        # valid_dict = {'text': [
        # "Since it was initiated by the Brazil workers' party~\cite{wainwright2003making} in the 90s, Participatory budgeting (PB)~\cite{cabannes2004participatory}"]}
        # val_ds = Dataset.from_dict(valid_dict)
        # return DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
        # drop_last=False,
        # collate_fn=partial(self.tokenize_batch))
        return self.dataloader_preprocessing('validation')


if __name__ == '__main__':
    dm = PretrainDataset(max_sequence_length=512)
    # make sure dataset has "training" key
    print(type(dm.dataset['train']))
    print(len(dm.dataset['train']))
    print(len(dm.dataset['train']))
    dataloader = dm.train_dataloader()
    for batch_id, batch in enumerate(dataloader):
        print(batch_id, batch)
        if batch_id == 3:
            break
