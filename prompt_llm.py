'''
To prompt the LLMs for 
1. Bad Question Generation 
2. Evaluating the Generated Questions  
'''
import os
import sys
import openai 
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError
import time
from utils import *

def set_api_key():
    '''
    Set OpenAI API key
    '''
    openai.api_key = os.getenv('OPENAI')


delay_time = 5
decay_rate = 0.8

def prompt_bad_question_generation(input_data):
    sys_prompt, fs_prompt = load_prompt()
    # print('### SYSTEM Prompt ###')
    # print(sys_prompt)
    # print('### Few Shot Prompt ###')
    # print(fs_prompt)
    input_prompt = fs_prompt + input_data + '\n'

    # construct message 
    messages = []
    messages.append({'role': 'system', 'content': sys_prompt})
    messages.append({'role': 'user', 'content': input_prompt})

    global delay_time

    # sleep to avoid rate limit error
    time.sleep(delay_time)

    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages,
            temperature=0.5,
            max_tokens=1000,
        )
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError) as exc:
        print(exc)
        delay_time *= 2
        return prompt_bad_question_generation(input_data)

    llm_response = response['choices'][0]['message']['content']

    # # save into temporary file for inspection 
    # with open('temp_llm_response.txt', 'w') as outfile:
    #     print(llm_response, file=outfile)

    return llm_response

def prompt_classification(input_data):
    sys_prompt, fs_prompt = load_classification_prompt()
    # print('### SYSTEM Prompt ###')
    # print(sys_prompt)
    # print('### Few Shot Prompt ###')
    # print(fs_prompt)
    input_prompt = fs_prompt + input_data + '\n'

    # construct message 
    messages = []
    messages.append({'role': 'system', 'content': sys_prompt})
    messages.append({'role': 'user', 'content': input_prompt})

    global delay_time

    # sleep to avoid rate limit error
    time.sleep(delay_time)

    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages,
            temperature=0.5,
            max_tokens=1000,
        )
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError) as exc:
        print(exc)
        delay_time *= 2
        return prompt_classification(input_data)

    llm_response = response['choices'][0]['message']['content']

    # # save into temporary file for inspection 
    # with open('temp_llm_response.txt', 'w') as outfile:
    #     print(llm_response, file=outfile)

    return llm_response



# if __name__ == '__main__':
#     prompt_bad_question_generation()