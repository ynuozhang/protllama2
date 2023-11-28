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
df1 = pd.read_csv(f'{params.input_csv_path}/ppigpt_train_merged_bernett_goldstandard_nov27_2023.csv')
df2 = pd.read_csv(f'{params.input_csv_path}/ppigpt_val_merged_bernett_goldstandard_nov27_2023.csv')
df3 = pd.read_csv(f'{params.input_csv_path}/ppigpt_test_merged_bernett_goldstandard_nov27_2023.csv')

seq1_train = df1['Sequence Interactor A'].values.tolist()
seq2_train = df1['Sequence Interactor B'].values.tolist()
seq1_valid = df2['Sequence Interactor A'].values.tolist()
seq2_valid = df2['Sequence Interactor B'].values.tolist()
seq1_test = df3['Sequence Interactor A'].values.tolist()
seq2_test = df3['Sequence Interactor B'].values.tolist()

train_dataset = Dataset.from_dict({'sequence_1': seq1_train,
                                   'sequence_2': seq2_train})
valid_dataset = Dataset.from_dict({'sequence_1': seq1_valid,
                                   'sequence_2': seq2_valid})
test_dataset = Dataset.from_dict({'sequence_1': seq1_test,
                                  'sequence_2': seq2_test})

dataset_dict = DatasetDict({
    'train': train_dataset,
    'valid': valid_dataset,
    'test': test_dataset
})

print(dataset_dict)
dataset_dict.save_to_disk(f'{params.output_dataset_path}/ppi_golden_raw.hf')
