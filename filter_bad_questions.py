'''
Filter generated bad questions based on the classification results
'''

import os 
import sys
from tqdm import tqdm
import json 
from collections import defaultdict
from classifiy_gen_questions import process_questions, parse_questions

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
    


if __name__ == '__main__':
    main()