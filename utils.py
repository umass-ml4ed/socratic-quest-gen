'''
utility functions
'''
from ast import literal_eval

def extract_text_in_tags(conversation_data, start_tag, end_tag):
    '''
    Extract the dialouge from the conversation data
    text between the tages - <dialogue> and <dialogue>
    '''
    # TODO: extract text between tags - <dialogue> and </dialogue>
    # find position of <dialogue>
    start_pos = conversation_data.find(start_tag)
    # find position of </dialogue>
    end_pos = conversation_data.find(end_tag)
    # extract text between tags
    dialouge = conversation_data[start_pos:end_pos].strip(start_tag).strip(end_tag).strip('\n')
    # return 
    return dialouge

def load_prompt():
    with open('prompt.txt', 'r') as f:
        prompt = f.read()
    # extract system message - between SYSTEM and INPUT 
    system_message = extract_text_in_tags(prompt, 'SYSTEM', 'INPUT').strip(':').strip()
    # start from the first INPUT
    few_shot_prompt = prompt[prompt.find('INPUT'):].strip('\n').strip()
    # return
    return system_message, few_shot_prompt

def load_classification_prompt():
    with open('classification_prompt.txt', 'r') as f:
        prompt = f.read()
    # extract system message - between SYSTEM and INPUT 
    system_message = extract_text_in_tags(prompt, 'SYSTEM', 'INPUT').strip(':').strip()
    # start from the first INPUT
    few_shot_prompt = prompt[prompt.find('INPUT'):].strip('\n').strip()
    # return
    return system_message, few_shot_prompt

def parse_question_output(ques_out):
    '''
    Converts a string of list into a python list
    '''
    # convert string to list
    ques_out_lst = literal_eval(ques_out)
    # return
    return ques_out_lst

