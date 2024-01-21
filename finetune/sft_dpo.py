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
from sklearn.model_selection import train_test_split
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
You are an Assistant whose task is to generate a "socratic" question to help the User debug their code. The generated question must not directly reveal the bug fix, and must be coherent with the conversation so far.
You are given the following inputs:
1. Problem Description and Test Cases (<problem>)
2. Student's buggy code (<bug_code>)
3. Bug Description (<bug_desc>)
4. Bug Fixes (<bug_fixes>)
5. Conversation (between User and Assistant) so far (<conversation>)

Metadata:
{}

<conversation>
{}'''.format(metadata, input_dialouge)
    
    return fix_prompt

def construct_data():
    '''
    create a list of dictionaries for SFT
    '''
    data_path = 'preference_data/'
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
        for ctr, dialouge in enumerate(input_dialouges):
            # construct prompt
            fix_prompt = construct_prompt(metadata, dialouge)
            # # fix_prompt = 'pseudo prompt change this'
            for good_output in good_outputs_list[ctr]:
                # append to all data
                all_data.append({'prompt': fix_prompt, 'output': good_output})
    
    return all_data

class QGSFTDataset(Dataset):
    '''
    QG Dataset
    '''
    def __init__(self, data: list):
        self.data = data
    
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class QGSFTCollator:
    def __init__(self, tokenizer, test: bool):
        self.tokenizer = tokenizer
        self.test = test

    def __call__(self, batch):
        all_prompts = [sample["prompt"] for sample in batch]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        if self.test:
            return {
                "input_ids": prompts_tokenized.input_ids.to(device),
                "attention_mask": prompts_tokenized.attention_mask.to(device),
                "meta_data": batch
            }

        # TODO: might be worth debugging this
        all_inputs = [sample["prompt"] + sample["output"] + self.tokenizer.eos_token for sample in batch]
        inputs_tokenized = self.tokenizer(all_inputs, return_tensors="pt", padding=True)
        prompt_lens = prompts_tokenized.attention_mask.sum(dim=1)
        labels = inputs_tokenized.input_ids.clone()
        padding_mask = torch.arange(labels.shape[1]).repeat(labels.shape[0], 1) < prompt_lens.unsqueeze(1)
        labels[padding_mask] = -100
        labels = labels.masked_fill(inputs_tokenized.attention_mask == 0, -100)
        return {
            "input_ids": inputs_tokenized.input_ids,
            "attention_mask": inputs_tokenized.attention_mask,
            "labels": labels
        }


######## TRAINING ########

def get_training_args(args):
    return TrainingArguments(
        output_dir=args.model_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        max_grad_norm=args.max_grad_norm or None,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_accumulation_steps=4,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else "none"
    )

def sft(args, train_data, val_data):
    assert args.model_name
    model, tokenizer = get_model(args.base_model, None, None, False, False)
    trainer = Trainer(
        model=model,
        args=get_training_args(args),
        train_dataset=QGSFTDataset(train_data),
        eval_dataset=QGSFTDataset(val_data),
        data_collator=QGSFTCollator(tokenizer, False)
    )
    trainer.train()
    trainer.save_model()

def add_params():
    parser = argparse.ArgumentParser()
    # Modes
    parser.add_argument("--sft", action="store_true", help="Supervised finetuning for feedback generation")
    parser.add_argument("--ppo", action="store_true", help="PPO training with reward model for feedback generation")
    parser.add_argument("--dpo", action="store_true", help="DPO training with GPT-4 annotations for feedback generation")
    parser.add_argument("--generate", action="store_true", help="Generate feedback with trained model")
    # Settings
    parser.add_argument("--base_model", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="Pre-trained base model path")
    parser.add_argument("--model_name", type=str, help="Name of model to save for training or load for testing")
    parser.add_argument("--pt_model_name", type=str, help="Name of pre-trained (SFT) model for RL training")
    parser.add_argument("--beta", type=float, default=0.1, help="KL regularization coefficient for DPO training")
    parser.add_argument("--mmo", type=int, default=1, help="Mismatch outer rate for DPO training")
    parser.add_argument("--decoding", type=str, choices=["greedy", "sample"], default="greedy", help="Decoding strategy for generation")
    parser.add_argument("--max_gen_tokens", type=int, default=128) # TODO: see what max size of question
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--wandb", action="store_true", help="Log performance to wandb")
    args = parser.parse_args()

    return args



def main():
    # add params
    args = add_params()

    # model, tokenizer = get_model('codellama/CodeLlama-7b-Instruct-hf', None, None, False, False)
    # construct data
    print('#### Constructing Data ####')
    all_data = construct_data()
    # split into train and val (80-20)
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=37)

    if args.wandb:
        print('NOT YET IMPLEMENTED')
    
    if args.sft:
        sft(args, train_data, val_data)


if __name__ == '__main__':
    main()
