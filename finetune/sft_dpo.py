'''
Perform standard fine-tuning and direct preference optimization (DPO) on the preference data.
'''

# imports 
from typing import Optional
import argparse
import os
import json
from itertools import combinations
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
from tqdm import tqdm
import wandb


######## MODEL ########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

peft_config = LoraConfig(
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    inference_mode=False,
)

def get_base_model(base_model_name: str, tokenizer: AutoTokenizer, test: bool):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=None if test else bnb_config,
        # Higher precision for non-quantized parameters helps training accuracy and doesn't hurt performance
        # Lower precision at test time improves speed and only marginally hurts performance
        torch_dtype=torch.float16 if test else torch.float32,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model

def get_model(base_model_name: str, model_name: Optional[str], pt_model_name: Optional[str],
              include_value_head: bool, test: bool, use_gradient_checkpointing: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = get_base_model(base_model_name, tokenizer, test)
    if test:
        # TODO: recommended to merge from quantized model - https://huggingface.co/docs/trl/main/en/dpo_trainer#downsides-to-merging-qlora-before-dpo-approach-2
        # If model was adapted on top of pre-trained model, load the pre-trained adapter first
        if pt_model_name:
            model = PeftModel.from_pretrained(model, pt_model_name).merge_and_unload()
        model = PeftModel.from_pretrained(model, model_name).merge_and_unload()
    else:
        # The pre-trained model serves as the base for the peft model AND as the reference model for KL regularization
        # The trl API can disable the adapters on the peft model to recover the reference model
        # Thus the reference model needs to be merged and unloaded, and doing it while quantized to save memory
        # Will cause "UserWarning: Merge lora module to 8-bit linear may get different generations due to rounding errors."
        if pt_model_name:
            model = PeftModel.from_pretrained(model, pt_model_name).merge_and_unload()
        # Create newly initialized LoRA adapters on top of base model
        # Gradient checkpointing can be used to save memory at cost of some time
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
        model = get_peft_model(model, peft_config)
        # Wrap model to add (newly initialized) value head for PPO training
        if include_value_head:
            model = AutoModelForCausalLMWithValueHead(model).to(device)
            model.is_peft_model = True # Tells PPO trainer to disable adapters to recover reference model
    return model, tokenizer




######## DATA ########
def construct_prompt(metadata, input_dialouge):
    '''
    return fixed prompt 
    '''

    fix_prompt = '''
    You are an Assistant whose task is to generate a "socratic" question to help the User debug their code.
    You are given the following inputs:
    1. Problem Description
    2. Test Cases
    3. Student's buggy code
    4. Bug Description 
    5. Bug Fixes
    6. Conversation (between User and Assistant) so far. 

    NOTE: 
    Analyze the student code, the bug description and fix and generate a "socratic" question to help the User debug their code.
    The generated question must not directly reveal the bug fix, and must be coherent with the conversation so far.

    Metadata:
    {}


    Conversation so far:
    {}
    '''.format(metadata, input_dialouge)
    
    return fix_prompt

def construct_data():
    '''
    create a list of dictionaries for SFT
    '''
    data_path = '../preference_data/'
    # input prompts 
    with open(os.path.join(data_path, 'input_prompts.json'), 'r') as infile:
        input_dialouges_dict = json.load(infile)
    # problem metadata
    with open(os.path.join(data_path, 'problem_metadata.json'), 'r') as infile:
        problem_metadata_dict = json.load(infile)
    # good data 
    with open(os.path.join(data_path, 'good_outputs.json'), 'r') as infile:
        good_outputs_dict = json.load(infile)
    

    all_data = []

    for tr_file, metadata in tqdm(problem_metadata_dict.items(), total=len(problem_metadata_dict)):
        input_dialouges = input_dialouges_dict[tr_file]
        good_outputs_list = good_outputs_dict[tr_file]
        for dialouge in input_dialouges:
            # construct prompt
            fix_prompt = construct_prompt(metadata, dialouge)
            for good_output in good_outputs_list:
                # append to all data
                all_data.append({'prompt': fix_prompt, 'output': good_output})
    
    return all_data


def main():
    model, tokenizer = get_model('codellama/CodeLlama-7b-Instruct-hf', None, None, False, False)
    # construct data
    print('#### Constructing Data ####')
    all_data = construct_data()
    print(all_data[0])


if __name__ == '__main__':
    main()
