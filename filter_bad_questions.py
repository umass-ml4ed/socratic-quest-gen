'''
Filter generated bad questions based on the classification results
'''

import os 
import sys
from tqdm import tqdm
from ast import literal_eval
import json 
from collections import defaultdict
from classifiy_gen_questions import process_questions, parse_questions

def check_validity(classification_result):
    '''
    Check if classification result is valid
    '''
    valid_categories = {'Irrelevant':0, 'Repeated':0, 'Direct':0, 'Premature':0}

    # extract string between curly braces ({})
    classification_result = classification_result[classification_result.find('{'):classification_result.find('}')+1].strip()
    classification_result_dict = literal_eval(classification_result)
    # sort the dict based on values 
    sorted_classification_result_dict = sorted(classification_result_dict.items(), key=lambda x: x[1], reverse=True)
    # extract the first key
    first_key = sorted_classification_result_dict[0][0]
    try:
        valid_categories[first_key] += 1
        decision = True
    except KeyError:
        decision = False
    return decision



def filter_bad_questions(bad_questions_list, classification_results_list):
    '''
    Filter bad questions based on classification results
    '''
    valid_bad_questions = []
    for question, classification_result in zip(bad_questions_list, classification_results_list):
        # TODO: Process classification result
        valid_result = check_validity(classification_result)
        if valid_result:
            valid_bad_questions.append(question)
    
    return valid_bad_questions


def main():
    data_path = 'socratic-debugging-benchmark/socratic_debugging_benchmark/v2_sigcse'
    train_path = os.path.join(data_path, 'train')

    # check if bad_questions.json exists
    if os.path.exists('bad_questions.json'):
        with open('bad_questions.json', 'r') as infile:
            bad_questions_dict = json.load(infile)
    else:
        print('Questions not generated')
        return

    # check if classification results already exists
    if os.path.exists('classification_results.json'):
        with open('classification_results.json', 'r') as infile:
            classification_results = json.load(infile)
    else:
        print('Classification results not generated')
        return
    
    
    # to store valid questions
    all_turn_results = defaultdict(list)
    for ctr, tr_file in tqdm(enumerate(os.listdir(train_path)), total=len(os.listdir(train_path)), desc='Processing files'):
        # bad questions for this file
        bad_questions_file = bad_questions_dict[tr_file]
        # classification results for this file
        classification_results_file = classification_results[tr_file]
        # assert length 
        assert len(bad_questions_file) == len(classification_results_file)
        # iterate over all turns 
        for turn in range(len(bad_questions_file)):
            bad_questions = process_questions([bad_questions_file[turn]])[0]
            classification_results_list = classification_results_file[turn]
            # parse bad questions
            bad_questions_list = parse_questions(bad_questions)
            # assert length
            assert len(bad_questions_list) == len(classification_results_list)
            # filter bad questions 
            valid_bad_questions = filter_bad_questions(bad_questions_list, classification_results_list)
            # print('###'*10)
            # print(valid_bad_questions)

            # store results 
            all_turn_results[tr_file].append(valid_bad_questions)

    # store valid bad questions
    with open('valid_bad_questions.json', 'w') as outfile:
        json.dump(all_turn_results, outfile, indent=4)
    


if __name__ == '__main__':
    main()