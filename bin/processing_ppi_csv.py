import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
import sentencepiece as spm
import os, sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='prepare PPI dataset')
    parser.add_argument('input_csv_path', type=str,
                        help='Path for the CSV dataframes')
    parser.add_argument('output_dataset_path', type=str,
                        help='Path for the processed raw dataset for downstream model utilization, should be Huggingface Datasets object')
    args = parser.parse_args()
    return args


params = parse_args()
df = pd.read_csv(f'{params.input_csv_path}/ppigpt_test_merged_MI0915_LTPHTP_8000AA_oct10_2023.csv')
df2 = pd.read_csv(f'{params.input_csv_path}/ppigpt_train_merged_MI0915_LTPHTP_8000AA_oct10_2023.csv')

seq1_valid = df['seq_1'].values.tolist()
seq2_valid = df['seq_2'].values.tolist()
seq1_train = df2['seq_1'].values.tolist()
seq2_train = df2['seq_2'].values.tolist()

train_dataset = Dataset.from_dict({'sequence_1': seq1_train,
                                   'sequence_2': seq2_train})
valid_dataset = Dataset.from_dict({'sequence_1': seq1_valid,
                                   'sequence_2': seq2_valid})

dataset_dict = DatasetDict({
    'train': train_dataset,
    'valid': valid_dataset
})
print(dataset_dict)
dataset_dict.save_to_disk(f'{params.output_dataset_path}/ppi_8000_raw.hf')
