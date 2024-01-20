'''
Creates a dataset of preferences from the raw data.
'''

import os
import json 
from tqdm import tqdm
from collections import defaultdict
from generate_bad_questions import seperate_turns, extract_text_in_tags

def process_assistant_turn(current_turn_good_outputs):
    '''
    split assistant turn using <alt> tabs
    '''
    current_turn_good_outputs = current_turn_good_outputs.strip('Assistant:')
    good_outputs = current_turn_good_outputs.split('<alt>')
    good_outputs = [x.strip() for x in good_outputs]
    return good_outputs

def create_preference_data(good_outputs_list, valid_bad_questions):
    '''
    create preference data
    '''
    preference_data = []
    for good_output in good_outputs_list:
        # iterate over valid bad questions
        for bad_question in valid_bad_questions:
            preference_data.append([good_output, bad_question])
    return preference_data

def construct_preference_data(turns, valid_bad_questions):
    '''
    construct the input prompts and preference data
    '''
    all_input_data = []
    all_preference_data = []
    for ctr, turn in enumerate(turns):
        # strip the Assistant turn
        current_turn_prompt = turn[:turn.find('Assistant:')].strip()
        # current turn good outputs
        current_turn_good_outputs = turn[turn.find('Assistant:'):].strip()
        # process good outputs 
        good_outputs_list = process_assistant_turn(current_turn_good_outputs)
        # construct preference data
        preference_data = create_preference_data(good_outputs_list, valid_bad_questions[ctr])

        # iterate over all previous turns 
        all_prev_turns = ''
        for prev_turn in turns[:ctr]:
            all_prev_turns += prev_turn 
        # construct input data
        input_data = all_prev_turns + current_turn_prompt + 'Assistant:\n'
        # append data
        all_input_data.append(input_data)
        all_preference_data.append(preference_data)
    
    return all_input_data, all_preference_data


def main():
    data_path = 'socratic-debugging-benchmark/socratic_debugging_benchmark/v2_sigcse'
    train_path = os.path.join(data_path, 'train')

    # load valid bad questions
    with open('valid_bad_questions.json', 'r') as infile:
        valid_bad_questions_dict = json.load(infile)
    
    # create storage dictionaries 
    all_input_prompts = defaultdict(list)
    all_preference_data = defaultdict(list)
    all_problem_metadata = defaultdict(list)

    # iterate over the train files 
    for ctr, tr_file in tqdm(enumerate(os.listdir(train_path)), total=len(os.listdir(train_path))):
        tr_file_path = os.path.join(train_path, tr_file)
        # valid bad questions for this file
        valid_bad_questions = valid_bad_questions_dict[tr_file]

        with open(tr_file_path, 'r') as f:
            conversation_data = f.read()
            # extract problem meta data - everything until </bug_fixes>
            problem_meta_data = conversation_data[:conversation_data.find('</bug_fixes>')+len('</bug_fixes>')].strip()
            # print(problem_meta_data)
            dialouge = extract_text_in_tags(conversation_data, '<dialogue>', '</dialogue>')
            # seperate turns
            turns = seperate_turns(dialouge)
            # construct preference data 
            input_prompts, preference_data = construct_preference_data(turns, valid_bad_questions)
            # add to storage dictionaries
            all_input_prompts[tr_file] = input_prompts
            all_preference_data[tr_file] = preference_data
            all_problem_metadata[tr_file] = problem_meta_data
    
    # save data into disk 
    save_dir = 'preference_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # save input prompts
    with open(os.path.join(save_dir, 'input_prompts.json'), 'w') as outfile:
        json.dump(all_input_prompts, outfile, indent=4)
    
    # save preference data
    with open(os.path.join(save_dir, 'preference_data.json'), 'w') as outfile:
        json.dump(all_preference_data, outfile, indent=4)
    
    # save problem metadata
    with open(os.path.join(save_dir, 'problem_metadata.json'), 'w') as outfile:
        json.dump(all_problem_metadata, outfile, indent=4)
 

if __name__ == '__main__':
    main()