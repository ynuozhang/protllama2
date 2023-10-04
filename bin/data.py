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
        print(f"Error during tokenization of string {s}: {e}")
        return []



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


class BatchedDataset(object):
    """inspired by esm2, but instead of sorting the original sequences,
    we should really sorting based on tokenized sequences
    """

    def __init__(self, sequence_strs, tokenizer, max_sequence_length):
        self.sequence_strs = sequence_strs['sequences']
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        # raw_tokenized_sequences = [self.tokenizer.encode(s) for s in self.sequence_strs]

        self.tokenized_sequences = []

        # automatically tokenize sequences upon creation of the object
        # if need manual, change it to call object.tokenize_sequence() in a separate line
        self.tokenize_sequences()

    def tokenize_sequences(self):
        checkpoint_interval = 1000000

        # any checkpoints?
        start_idx, file_name = self.get_latest_checkpoint_sequence()

        if start_idx > 0:
            with open(file_name, 'rb') as f:
                self.tokenized_sequences = pickle.load(f)
                print(f'Resume training from {start_idx}')

        with Pool(processes=64, initializer=init_pool, initargs=(self.tokenizer,)) as pool:
            for idx, result in enumerate(
                    tqdm(pool.imap(partial(standalone_tokenize_function, max_sequence_length=self.max_sequence_length),
                                   self.sequence_strs[start_idx:]),
                         total=len(self.sequence_strs) - start_idx)):
                self.tokenized_sequences.extend(result)
                if (idx + start_idx) % checkpoint_interval == 0:
                    self.save_checkpoint(idx + start_idx)

    def save_checkpoint(self, idx):
        print(f'Start generating tokens for {idx} strs')
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with open(f'intermediate_checkpoint_{idx}_{current_time}.pkl', 'wb') as f:
            pickle.dump(self.tokenized_sequences, f)
        print(f'Finish generating tokens for {idx} strs')

    @staticmethod
    def get_latest_checkpoint_sequence(ckpt_path='/data/rozen/home/e0833634/lama/protllama/batch_script/*.pkl'):
        cache_list = glob.glob(ckpt_path)
        if cache_list:
            cache_fname = max(cache_list, key=os.path.getctime)
            return int(cache_fname.split('_')[3]), cache_fname
        else:
            return 0

        # def tokenize_function(self, s):
        # tokens = self.tokenizer.encode(s)
        # tokenized_sequence = []
        # if len(tokens) >= self.max_sequence_length - 1:
        # considering the added special token bos
        # sampled_windows = self.sample_windows(tokens, self.max_sequence_length)
        # for sample in sampled_windows:
        # sample.insert(0, self.tokenizer.bos_id())
        # tokenized_sequence.append(sample)
        # else:
        # tokens.insert(0, self.tokenizer.bos_id())
        # tokenized_sequence.append(tokens)
        # return tokenized_sequence

        # @staticmethod
        # def sample_windows(sequence, max_sequence_length, extra_toks_per_seq=1):
        # import random
        # random.seed(42)
        #"""based on Phil's implement
        #Returns: a list of sampled windows.
        #"""
        # sampled_windows = []
        # the beginning window
        # sampled_windows.append(sequence[:max_sequence_length - extra_toks_per_seq])
        # calculate the num of random slices needed, remove head and tail
        # num_slices_required = (len(sequence) // max_sequence_length) - 2
        # max_start_index = len(sequence) - max_sequence_length

        # if max_start_index < 0:
        # raise ValueError("max_sequence_length greater than length of sequence")

        # if num_slices_required > 0:
        # for _ in range(num_slices_required):
        # Randomly select start index of the window
        # start_index = random.randint(0, max_start_index)
        # Extract window
        # Considering the added special token bos
        # window = sequence[start_index:start_index + max_sequence_length - extra_toks_per_seq]
        # Append the window to the list of sampled windows
        # sampled_windows.append(window)

        # the end window
        # sampled_windows.append(sequence[-(max_sequence_length - extra_toks_per_seq):])
        # return sampled_windows

    def get_batch_indices(self, extra_toks_per_seq=1):
        sizes = [(len(tokens), i) for i, tokens in enumerate(self.tokenized_sequences)]
        sizes.sort()
        print(sizes)
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
        return len(self.tokenized_sequences)

    def __getitem__(self, idx):
        return self.tokenized_sequences[idx]


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


class pretrainDataset(pl.LightningDataModule):
    def __init__(self,
                 # batch_size: int = 1,
                 target: str = 'protein',
                 max_sequence_length: int = 512):
        super().__init__()
        self.original_data = None
        self.dataset_path = '/data/rozen/home/e0833634/lama/data/swiss_2023_9/uniref50_random90split_Dataset.hf'
        self.batch_path = '/data/rozen/home/e0833634/lama/data/swiss_2023_9/uniref50_random90split'
        self.vocab_size = '8k'
        self.target = target
        self.tokenizer = self.tokenizer_generation(self.target, self.vocab_size)
        self.max_sequence_length = max_sequence_length

        if not os.path.exists(self.dataset_path):
            print('Start generating tokenized datasets')
            self.save_tokenized_data()
            assert hasattr(self, 'dataset')
            assert hasattr(self, 'batches')
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

    def process_and_store_data(self):
        self.original_data = self.retrieve_data(self.target)
        self.tokenized_data = {}
        tokenize_batch_fn = TokenizeBatch(self.tokenizer)
        for split_name, ds in self.original_data.items():
            batched_dataset = BatchedDataset(ds, self.tokenizer, self.max_sequence_length)
            batches = batched_dataset.get_batch_indices()

            all_input_ids, all_attention_masks, all_labels = [], [], []
            for batch_idx in batches:
                batch = [batched_dataset[i] for i in batch_idx]
                tokenized_batch = tokenize_batch_fn(batch)

                all_input_ids.append(tokenized_batch['input_ids'])
                all_attention_masks.append(tokenized_batch['attention_mask'])
                all_labels.append(tokenized_batch['labels'])

            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_attention_masks = torch.cat(all_attention_masks, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            self.tokenized_data[split_name] = {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_masks,
                'labels': all_labels
            }
            with open(self.batch_path + '_' + split_name + '_Batches.pkl', "wb") as file:
                pickle.dump(batches, file)
            print('finish generating dataloader for ', split_name)

    def save_tokenized_data(self):
        self.process_and_store_data()
        self.dataset = DatasetDict({
            'train': Dataset.from_dict(self.tokenized_data['train']),
            'validation': Dataset.from_dict(self.tokenized_data['valid'])
        })

        self.dataset.save_to_disk(self.dataset_path)
        # with open(self.dataset_path, "wb") as file:
        # pickle.dump(self.dataset, file)

    def dataloader_preprocessing(self, split_name):
        ds = self.dataset[split_name]
        with open(self.batch_path + '_' + split_name + '_Batches.pkl', 'rb') as file:
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
    dm = pretrainDataset(max_sequence_length=512)
    # make sure dataset has "training" key
    print(type(dm.dataset['train']))
    print(len(dm.dataset['train']))
    print(len(dm.dataset['train']))
    dataloader = dm.train_dataloader()
    for batch_id, batch in enumerate(dataloader):
        print(batch_id, batch)
        if batch_id == 3:
            break
