'''
conduct zero-shot LLama Chat prompting for socratic guidance generation
'''

import os 
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch 
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

def construct_prompt(metadata, input_dialouge):
    '''
    return fixed prompt 
    '''

    fix_prompt = '''
[INST] Generate "socratic" guidance as a "Socratic Guiding Assistant" to help the User debug their code. The generated guidance must help the User realize their bug. The guidance must not directly reveal the bug fix, and must be coherent with the conversation so far.
You are given the following metadata:
1. Problem Description and Test Cases (<problem>)
2. Student's buggy code (<bug_code>)
3. Bug Description (<bug_desc>)
4. Bug Fixes (<bug_fixes>)
5. Conversation (between User and Assistant) so far (<CONVERSATION>)[/INST]

<METADATA>
{}
</METADATA>

<CONVERSATION>
{}

NOTE: Your output MUST BE A PYTHON LIST (enclosed in square bracketts) of socratic guidance ["write your first guidance here", "your second guidance goes here", "your third one goes here", "and so on..."]

Remeber to enclose the string in double quotes and not yse any unicode characters.

Output:'''.format(metadata, input_dialouge)
    
    return fix_prompt

def construct_prompt_single(metadata, input_dialouge):
    '''
    return fixed prompt 
    '''

    fix_prompt = '''
[INST] Generate "socratic" guidance as a "Socratic Guiding Assistant" to help the User debug their code. The generated guidance must help the User realize their bug. The guidance must not directly reveal the bug fix, and must be coherent with the conversation so far.
You are given the following metadata:
1. Problem Description and Test Cases (<problem>)
2. Student's buggy code (<bug_code>)
3. Bug Description (<bug_desc>)
4. Bug Fixes (<bug_fixes>)
5. Conversation (between User and Assistant) so far (<CONVERSATION>)[/INST]

<METADATA>
{}
</METADATA>

<CONVERSATION>
{}

Generate a single socratic guidance.

Output:'''.format(metadata, input_dialouge)
    
    return fix_prompt

def construct_prompt_cot(metadata, input_dialouge):
    '''
    return fixed prompt 
    '''

    fix_prompt = '''
[INST] Generate "socratic" guidance as a "Socratic Guiding Assistant" to help the User debug their code. The generated guidance must help the User realize their bug. The guidance must not directly reveal the bug fix, and must be coherent with the conversation so far.
You are given the following metadata:
1. Problem Description and Test Cases (<problem>)
2. Student's buggy code (<bug_code>)
3. Bug Description (<bug_desc>)
4. Bug Fixes (<bug_fixes>)
5. Conversation (between User and Assistant) so far (<CONVERSATION>)[/INST]

<METADATA>
{}
</METADATA>

<CONVERSATION>
{}

NOTE: 
1. First generate a set of misconceptions that the user still has based on the Conversation so far. 
2. Your output MUST BE TWO PYTHON LISTS (enclosed in square bracketts) of misconcpetions and socratic guidance 
Example:
Output:
Misconceptions: ['misconception 1', 'misconception 2', 'misconception 3']
Guidance: ['write your first guidance here', 'your second guidance goes here', 'your third one goes here', 'and so on...']

Remeber to enclose the string in double quotes and not yse any unicode characters.

Output:'''.format(metadata, input_dialouge)
    
    return fix_prompt


def generate_completion(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device) -> str:
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = inputs.size()[1]
    max_length = input_length + 500
    output = model.generate(inputs, max_length=max_length, num_return_sequences=1, do_sample=False, top_p=0)
    completion = tokenizer.decode(output[:, input_length:][0], skip_special_tokens=True)
    return completion
 

def load_llama_model():
    '''
    loads and returns the llama model
    '''
    llama_2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    llama_2_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    llama_2_tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available')
        return

    print("Using device:", device)
    print("Sending model to device this might take a while.")
    llama_2_model.to(device)

    return llama_2_model, llama_2_tokenizer


def construct_data(split_path: str, prompt_type: str):
    '''
    create a list of dictionaries for SFT
    '''
    data_path = os.path.join('preference_data', split_path)
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
            if prompt_type == 'single':
                fix_prompt = construct_prompt_single(metadata, dialouge)
            elif prompt_type == 'five':
                fix_prompt = construct_prompt(metadata, dialouge)
            elif prompt_type == 'cot':
                fix_prompt = construct_prompt_cot(metadata, dialouge)
            all_data.append({'prompt': fix_prompt, 'output': str(good_outputs_list[ctr])})
    return all_data

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, choices=["single", "five", "cot"], default="single", help="number of guidance to generate")
    args = parser.parse_args()
    return args


def main():
    # add params
    args = add_params()

    # laod llama model
    llama_2_model, llama_2_tokenizer = load_llama_model()
    # get dataset
    all_data = construct_data('testset', args.prompt)
    # iterate over the data 
    print('Generating completions')
    for data in tqdm(all_data, total=len(all_data)):
        prompt = data['prompt']
        # generate completion
        completion = generate_completion(prompt, llama_2_model, llama_2_tokenizer, torch.device("cuda"))
        # add result to the data
        data['prediction'] = completion
    
    # convert data into a dataframe
    df = pd.DataFrame(all_data)
    # save the dataframe
    df.to_csv('results/llama_zero_shot_results_{}.csv'.format(args.prompt), index=False)
        

if __name__ == "__main__":
    main()