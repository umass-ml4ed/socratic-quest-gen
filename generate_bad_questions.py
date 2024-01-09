'''
Use OpenAI GPT to generate bad questions
'''
import os 
import json
from collections import defaultdict
from utils import *
from tqdm import tqdm
from prompt_llm import *

def seperate_turns(dialouge):
    '''
    Seperate the dialouge into turns
    '''
    # iterate over the dialouge
    collect_turns = []
    clear_line = ''
    for line in dialouge.split('\n'):
        if 'User:' in line:
            collect_turns.append(clear_line)
            clear_line = ''
        clear_line += line
    # print(collect_turns)
    # remove first element - empty string
    collect_turns.pop(0)
    # return
    return collect_turns

def construct_input_data(turns):
    '''
    construct data for inputting to the question generator
    '''
    all_input_data = []
    for ctr, turn in enumerate(turns):
        # strip the Assistant turn
        current_turn = turn[:turn.find('Assistant:')].strip()
        # iterate over all previous turns 
        all_prev_turns = ''
        for prev_turn in turns[:ctr]:
            all_prev_turns += prev_turn 
        # construct input data
        input_data = all_prev_turns + current_turn + 'Assistant:\n'
        all_input_data.append(input_data)
    
    return all_input_data


def main():

    # set api key
    set_api_key()


    data_path = 'socratic-debugging-benchmark/socratic_debugging_benchmark/v2_sigcse'
    train_path = os.path.join(data_path, 'train')

    store_results = defaultdict(list)

    for ctr, tr_file in tqdm(enumerate(os.listdir(train_path)), total=len(os.listdir(train_path))):
        tr_file_path = os.path.join(train_path, tr_file)
        with open(tr_file_path, 'r') as f:
            conversation_data = f.read()
            # extract problem meta data - everything until </bug_fixes>
            problem_meta_data = conversation_data[:conversation_data.find('</bug_fixes>')+len('</bug_fixes>')].strip()
            # print(problem_meta_data)
            dialouge = extract_text_in_tags(conversation_data, '<dialogue>', '</dialogue>')
            # seperate turns
            turns = seperate_turns(dialouge)
            # print(turns)
            # construct input data
            all_input_conversation = construct_input_data(turns)
            # print('#### Input Data ####')
            # print(all_input_conversation[2])

            for cctr, conversation in enumerate(all_input_conversation):
                # construct input prompt 
                input_prompt = problem_meta_data + '\n\n<dialogue>' + conversation + '\nOUTPUT:\n'
                print('#### Input Prompt ####')
                print(input_prompt)

                # generate bad question
                llm_response = prompt_bad_question_generation(input_prompt)
                print('#### LLM Response ####')
                print(llm_response)
                store_results[tr_file].append(llm_response)
        break
    
    # save results
    with open('bad_questions.json', 'w') as outfile:
        json.dump(store_results, outfile, indent=6)

if __name__ == '__main__':
    main()