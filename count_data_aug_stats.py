'''
compute data augmentation statistics
'''

import os
import json
from ast import literal_eval

def main():
    # load bad_questions.json
    with open('bad_questions.json') as f:
        bad_questions = json.load(f)
    
    # load valid_bad_questions.json
    with open('valid_bad_questions.json') as f:
        valid_bad_questions = json.load(f)
    
    # total bad_questions
    total_bad_questions = 0
    total_valid_ques = 0
    for file, questions in bad_questions.items():
        total_bad_questions += (len(questions)*4)
        valid_bad_ques_list = valid_bad_questions[file]
        # count length of list of lists
        total_valid_ques += sum([len(lst) for lst in valid_bad_ques_list])
    
    print('Total bad questions: ', total_bad_questions)
    print('Total valid questions: ', total_valid_ques)

    # count classificatin results
    with open('classification_results.json') as f:
        classification_results = json.load(f)
    
    # iterate over data
    total_good = 0
    total_incorrect = 0
    total_questions = 0
    for file, all_results in classification_results.items():
        for results in all_results:
            for res in results:
                total_questions += 1
                res = res[res.find('{'):res.find('}')+1].strip()
                res_dict = literal_eval(res)
                if res_dict['Good'] == 1: 
                    total_good += res_dict['Good']
                if res_dict['Incorrect'] == 1:
                    total_incorrect += res_dict['Incorrect']
    
    print('\n\nClassification Results')
    print('Total Good: ', total_good)
    print('Total Incorrect: ', total_incorrect)
    print('Total Questions: ', total_questions)


if __name__ == '__main__':
    main()