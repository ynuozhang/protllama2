"""ppi llama inference code, written on Nov 27"""
import torch
import sys
import numpy as np
from model import pretrainLlama
from argparse import Namespace
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from lightning import Trainer, seed_everything
seed_everything(42)

checkpoint = torch.load('/data/rozen/home/e0833634/lama/protllama/pl_model_cache/epoch=23-train_perplexity=1.161-val_perplexity=255.593-ppi_10_26_10k_2048.ckpt')
hyper_parameters = checkpoint["hyper_parameters"]

# Assuming you have the original Namespace object
original_hparam = hyper_parameters['hparam']

# Create a new Namespace object with the updated tokenizer_path
new_hparam = Namespace(
    accumulate_grad_batches=original_hparam.accumulate_grad_batches,
    attempts=original_hparam.attempts,
    batch_size=original_hparam.batch_size,
    date=original_hparam.date,
    devices=original_hparam.devices,
    epoch=original_hparam.epoch,
    flash_attention=original_hparam.flash_attention,
    hidden_size=original_hparam.hidden_size,
    input_dataset_path=original_hparam.input_dataset_path,
    intermediate_size=original_hparam.intermediate_size,
    learning_rate=original_hparam.learning_rate,
    max_position_embeddings=original_hparam.max_position_embeddings,
    num_attention_heads=original_hparam.num_attention_heads,
    num_hidden_layers=original_hparam.num_hidden_layers,
    num_key_value_heads=original_hparam.num_key_value_heads,
    num_workers=original_hparam.num_workers,
    output_dataset_path=original_hparam.output_dataset_path,
    save_top_k=original_hparam.save_top_k,
    scheduler=original_hparam.scheduler,
    strategy=original_hparam.strategy,
    target=original_hparam.target,
    tokenizer_path='/data/rozen/home/e0833634/lama/protllama/batch_script/',  # Update the tokenizer_path here
    train_dataloader_length=original_hparam.train_dataloader_length,
    vocab_size=original_hparam.vocab_size
)

# Update the hyper_parameters with the new Namespace
hyper_parameters['hparam'] = new_hparam
model = pretrainLlama(**hyper_parameters)
model.configure_model()
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

#model.model.generation_config.update({"max_length":100})

tokenizer = LlamaTokenizerFast('/data/rozen/home/e0833634/lama/protllama/batch_script/protein_10k.model')
prompt = "YAPSALVLTVGKGVSATTAAPERAVTLTCAPGPSGTHPAAGSACADLAAVGGDLNALTRGEDVMCPMVYDPVLLTVDGVWQGKRVSYERVFSNECEMNAHGSSVFAF</s>"
inputs = tokenizer(prompt, return_tensors="pt")

tokenizer.pad_token_id = tokenizer.eos_token_id
model.model.generation_config.pad_token_id = model.model.generation_config.eos_token_id
model.model.generation_config.max_length = 200
# Parameters for manipulation of the model output logits
model.model.generation_config.repetition_penalty = 1.2
model.model.generation_config.temperature = 0.7 # randomness
#model.model.generation_config.top_p = 0.9
model.model.generation_config.top_k = 950
model.model.generation_config.length_penalty = -0.1
# model.model.generation_config.length_penalty = 1 # < 0 encourage shorter sequences
# Parameters that control the generation strategy used
model.model.generation_config.do_sample = True # use greedy decoding otherwise
# Parameters that define the output variables of `generate`
model.model.generation_config.num_return_sequences = 2
model.model.generation_config.output_scores = True
model.model.generation_config.return_dict_in_generate = True

for key in inputs:
    inputs[key] = inputs[key].to('cuda')
# Define the custom prefix_allowed_tokens function to enforce conditioning on A
model = model.cuda()
with torch.cuda.amp.autocast():
    # Generate text B conditioned on A
    generated_output =model.model.generate(inputs.input_ids,
                                           #prefix_allowed_tokens_fn=custom_prefix_allowed_tokens,
                                           )

#works only when you generate one output
def calculate_single_perplexity(output_scores, transition_scores):
    #intializing stuff
    #num_tokens_generated = len(output_scores)
    logits_for_every_tok = torch.stack(output_scores) # [sequence length, vocab size]
    specific_logits = torch.tensor(transition_scores) # [1, sequence length]
    # tile specific_logits to match the shape of logits_for_every_tok in second dimension
    logits_for_every_tok_transposed = logits_for_every_tok.transpose(0, 1) # [vocab size, sequence length]
    specific_logits = specific_logits.squeeze(0) # double check to make sure its shaped in [, sequence length]
    close_mask = torch.isclose(logits_for_every_tok_transposed, specific_logits, rtol=1e-5)
    # returns torch matrix indices like [0, 0], indicating row 0 column 0, close_indices has size of [sequence length, 2]
    # indicating the chosen logits per position

    # find the indices of non-zero elements along axis 0
    nonzero_indices = torch.nonzero(close_mask, as_tuple=False)
    # sort the indices based on the first element (axis 0)
    sorted_indices = nonzero_indices[nonzero_indices[:, 0].argsort()]
    # get the unique indices along axis 1
    unique_indices = torch.unique(sorted_indices[:, 1], sorted=False)

    # only keep the first one that matched along axis 0, meaning that only keep the first vocab that hit the isclos() function
    filtered_close_indices = torch.stack([sorted_indices[sorted_indices[:, 1] == i][0] for i in unique_indices])

    # locate logits that are not '-inf', and then do exp, which will be used as the denominator
    mask = (logits_for_every_tok_transposed != -float('inf'))
    tensor_exp = torch.exp(logits_for_every_tok_transposed * mask)
    tensor_without_nan = torch.nan_to_num(tensor_exp, nan=0.0)

    log_prob = 0.0
    for i in range(logits_for_every_tok_transposed.size(1)):
        # locate the logit that was close to the transition_scores
        softmax_numerator = np.exp(logits_for_every_tok_transposed[filtered_close_indices[i][0].item(), # first axis
                                   filtered_close_indices[i][1].item()]) # second axis
        softmax_denominator = torch.sum(tensor_without_nan[:, i])
        probability = softmax_numerator / softmax_denominator
        log_prob += np.log(probability.item())
    ppl = np.exp(-log_prob / logits_for_every_tok_transposed.size(1))
    return ppl

    #logits_for_every_tok = []
    #for i in range(num_tokens_generated):
        #logits_for_every_tok.append(output_scores[i]) #only one output, so 0
    # initialize a list to hold the indices
    #specific_logits = transition_scores.tolist()
    #indices = []
    # iterate through each sub-list and the specific logit at the same time
    #for i, (sub_list, specific_logit) in enumerate(zip(logits_for_every_tok, specific_logits)):
        # find index of specific logit in the sub-list & use small epsilon for float comparison - round errors
        #epsilon = 1e-5
        #array = sub_list.numpy()
        #index = next((idx for idx, val in enumerate(sub_list) if abs(val - specific_logit) < epsilon), None)
        #indices.append(index)
    #'indices' now contains the indices of each specific logit in the corresponding sub-list

   # log_prob = 0.0
    #for i in range(num_tokens_generated):
        #get rid of -inf for numerical stabulity
        #logits = [float(k) for k in output_scores[i] if k !=float('-inf')] #btw, you might want to know that len(logits) = top_k
        # Apply the softmax function to convert logits to probabilities
        #total_sum = np.exp(logits).sum()
        #probability = np.exp(output_scores[i][indices[i]])/total_sum
        #ppl = e^(-(average of log probability)) #https://huggingface.co/docs/transformers/perplexity
        #log_prob+=np.log(probability.item())
    #ppl = np.exp(-log_prob/len(indices))

   # return ppl

def calculate_multiple_perplexity(num_outputs, outputs_scores, transition_scores):
    perplexities = []
    for i in range(num_outputs):
        tran_score = transition_scores[i, :]
        out_score = []
        for k in outputs_scores:
            out_score.append(k[i].cpu())
        ppl = calculate_single_perplexity(out_score, tran_score)
        perplexities.append(ppl)
    return perplexities

#calculate ppl
transition_scores = model.model.compute_transition_scores(
    generated_output.sequences, generated_output.scores, normalize_logits=False #normalize_logits applies softmax then log
    )
transition_scores_np = np.array(transition_scores.cpu())

perplexity = calculate_multiple_perplexity(model.model.generation_config.num_return_sequences,
                                           generated_output.scores,
                                           transition_scores_np)
top_k_text = tokenizer.batch_decode(generated_output['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]