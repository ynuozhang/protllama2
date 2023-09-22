import gc
from tqdm import tqdm
import pickle
from tokenizers import Tokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from functools import partial
from typing import Optional, Tuple
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from data_utils_modified import prepare_dataloader_modified
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import Dataset

gc.collect()
torch.cuda.empty_cache()


def load_tokenizer_with_special_tokens(path):
    # Load the tokenizer
    tokenizer = Tokenizer.from_file(path + 'tokenizer.json')

    # Set bos_token_id and eos_token_id to 0
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0

    return tokenizer


def tokenize_batch(batch, tokenizer: Optional[LlamaTokenizer] = None, max_length: int = 2048):
    texts = [sample['text'] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    data['labels'] = data['input_ids'].clone()
    return data


# initialize config for the model

MODEL_CONFIGS = {
    '7b': LlamaConfig(max_position_embeddings=512,
                      hidden_size=640,
                      intermediate_size=1720)}

with open('/data/rozen/home/e0833634/lama/protllama/original_lama.pkl', 'rb') as f:
    loaded_data_lama = pickle.load(f)

train_ds = loaded_data_lama['train']

valid_dict = {'text': [
    "Since it was initiated by the Brazil workers' party~\cite{wainwright2003making} in the 90s, Participatory budgeting (PB)~\cite{cabannes2004participatory}"]}
valid_dataset = Dataset.from_dict(valid_dict)

tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
tokenizer.pad_token = tokenizer.unk_token

dataloader = prepare_dataloader_modified(train_ds, batch_size=1, shuffle=False,
                                         drop_last=True,
                                         collate_fn=partial(tokenize_batch, tokenizer=tokenizer, max_length=1024))
valid_dataloader = prepare_dataloader_modified(valid_dataset, batch_size=1, shuffle=False,
                                               drop_last=True,
                                               collate_fn=partial(tokenize_batch, tokenizer=tokenizer, max_length=1024))
start_epoch = 0
start_step = 0
sampler_start_idx = 0
num_steps_per_epoch = len(dataloader)
num_epoch = 1

torch.manual_seed(42)
import random

random.seed(42)
config = '7b'
config = MODEL_CONFIGS[config]
model = LlamaForCausalLM(config)
print(model.lm_head.weight)

optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=100,
                                            num_training_steps=num_epoch * len(dataloader))
device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(start_epoch, num_epoch):
    model.train()
    with tqdm(enumerate(dataloader),
              desc=f'Epoch {epoch}',
              total=num_steps_per_epoch,
              initial=start_step) as pbar:
        for step, batch in pbar:
            optimizer.zero_grad()
            for k, v in batch.items():
                batch = {k: v.cuda() for k, v in batch.items()}
                break
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            break

    model.eval()
    with tqdm(enumerate(valid_dataloader),
              desc=f'Epoch {epoch}',
              initial=start_step) as pbar:
        with torch.no_grad():
            for step, batch_valid in enumerate(valid_dataloader):
                for k_, v_ in batch_valid.items():
                    batch_valid = {k_: v_.cuda() for k_, v_ in batch_valid.items()}
                outputs_valid = model(**batch_valid)
                loss_valid = outputs_valid[0].item()
print('training loss: \n', loss, 'validation loss: ', loss_valid)
