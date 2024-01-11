'''
Classifies generated bad questions using an LLM
'''
import os
import sys
import re
import copy
from tqdm import tqdm
from collections import defaultdict
import json
from ast import literal_eval
from utils import *
from generate_bad_questions import *
from prompt_llm import *

def parse_questions(bad_questions):
    '''
    parse all text occuring after 'Question' in the bad_questions
    '''
    # # Define the regular expression pattern
    # # Define the regular expression pattern with re.DOTALL to match across multiple lines
    # pattern = r"'Question': '(.*?)'"

    # # Use re.findall to extract questions
    # questions = re.findall(pattern, bad_questions, re.DOTALL)
    data_str = bad_questions.replace('"', '')
    data_str = re.sub("([A-Za-z])'([A-Za-z])", r"\1\'\2", data_str)
    corrected_data_str = data_str.replace("'", '"')
    data = json.loads(corrected_data_str)
    questions = [item['Question'] for item in data]

    return questions

def construct_input_data(turns, all_bad_questions):
    '''
    remove assistant converstaion at the end of each turn
    create input data by adding bad_questions
    '''
    all_input_data = []
    for ctr, (turn, bad_questions) in enumerate(zip(turns, all_bad_questions)):
        # strip the Assistant turn
        current_turn = turn[:turn.find('Assistant:')].strip()
        # iterate over all previous turns 
        all_prev_turns = ''
        for prev_turn in turns[:ctr]:
            all_prev_turns += prev_turn 
        # construct input data
        input_data = all_prev_turns + current_turn + 'Assistant Socratic Question: '
        # TODO: add bad questions
        # parse bad questions 
        cat_wise_inp_data = [] 

        # bad_questions = literal_eval(bad_questions) # literal_eval not working
        try:
            clean_bad_questions = parse_questions(bad_questions)
        except json.decoder.JSONDecodeError:
            print(bad_questions)
            all_input_data.append(None) # add None to indicate that this example is not valid
        # print(bad_questions)
        # print(clean_bad_questions)
        # sys.exit(0)
        for quest_data in clean_bad_questions:
            # copy input data into a separate string variable 
            input_data_copy = copy.copy(input_data)
            input_data_copy += quest_data + '\n'
            cat_wise_inp_data.append(input_data_copy)
            del input_data_copy
        # print('#### Cat Wise Input Data ####')
        # print(cat_wise_inp_data)
        # if ctr == 1:
        #     sys.exit(0)

        all_input_data.append(cat_wise_inp_data)
    return all_input_data

def process_questions(bad_questions):
    '''
    remove logic output
    '''
    # # print('In Process Questions\n')
    # Remove 'Logic' parts using regular expression
    new_bad_questions = []
    for quest in bad_questions:
        output_string = re.sub(r"'Logic':.*?'Question'", "'Question'", quest)
        # print(output_string)
        new_bad_questions.append(output_string)
    return new_bad_questions



def main():
    system_message, few_shot_prompt = load_classification_prompt()
    # print('### System Message ###')
    # print(system_message)
    # print('### Few Shot Prompt ###')
    # print(few_shot_prompt)

    data_path = 'socratic-debugging-benchmark/socratic_debugging_benchmark/v2_sigcse'
    train_path = os.path.join(data_path, 'train')

    store_results = defaultdict(list)

    # check if results file already exists
    if os.path.exists('bad_questions.json'):
        with open('bad_questions.json', 'r') as infile:
            store_results = json.load(infile)
    else:
        print('Questions not generated')
        return
    
    for ctr, tr_file in tqdm(enumerate(os.listdir(train_path)), total=len(os.listdir(train_path))):
        print(tr_file)
        tr_file_path = os.path.join(train_path, tr_file)
        with open(tr_file_path, 'r') as f:
            conversation_data = f.read()
            # extract problem meta data - everything until </bug_fixes>
            problem_meta_data = conversation_data[:conversation_data.find('</bug_fixes>')+len('</bug_fixes>')].strip()
            # print(problem_meta_data)
            dialouge = extract_text_in_tags(conversation_data, '<dialogue>', '</dialogue>')
            # seperate turns
            turns = seperate_turns(dialouge)
            # load bad_questions
            bad_questions = process_questions(store_results[tr_file])
            # check if length if equal 
            assert len(turns) == len(bad_questions)
            # construct input data
            all_input_conversation = construct_input_data(turns, bad_questions) # list of list of strings
            # print('#### Input Data ####')
            # print(all_input_conversation)

            # # print stats
            # print('Number of turns: ', len(turns))
            # print('Number of bad questions: ', len(bad_questions))
            # print('Number of input conversations: ', len(all_input_conversation))
            # print('Number of examples within each input conversation: ', len(all_input_conversation[0]))
            # print('Sample Last Conversation: ', all_input_conversation[-1][-1])
        break


if __name__ == '__main__':
    main()