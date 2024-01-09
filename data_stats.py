'''
Explore the number of turns and total positive traning examples
'''
import os
from tqdm import tqdm
from utils import *


def count_dialouge_stats(dialouge):
    '''
    counts number of positive training samples
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
    # assistant turns
    assis_turns = len(collect_turns)
    # iterate over each conversation turn
    tot_alt_count = 0
    for turn in collect_turns:
        # extract text starting from Assistant:
        assis_text = turn[turn.find('Assistant:')+len('Assistant:'):].strip()
        # count number of <alt> tags
        alt_count = assis_text.count('<alt>')
        tot_alt_count += alt_count
    
    # print('Assistant turns: ', assis_turns)
    # print('Total alt tags: ', tot_alt_count)
        
    return assis_turns, tot_alt_count

def retrive_stats(data_path):
    # iterate over train files
    all_asis_turns, all_alt_count = [], []
    for ctr, tr_file in tqdm(enumerate(os.listdir(data_path)), total=len(os.listdir(data_path))):
        tr_file_path = os.path.join(data_path, tr_file)
        with open(tr_file_path, 'r') as f:
            conversation_data = f.read()
        # TODO: extract text between tags - <dialogue> and </dialogue>
        dialouge = extract_text_in_tags(conversation_data, '<dialogue>', '</dialogue>')
        # count stats
        assis_turns, tot_alt_count = count_dialouge_stats(dialouge)
        all_asis_turns.append(assis_turns)
        all_alt_count.append(tot_alt_count)
    
    # print stats
    print('Total training dialouges: ', ctr+1)
    print('Total assistant turns: ', sum(all_asis_turns))
    print('Total alt tags: ', sum(all_alt_count))
    print('All Positive Examples', sum(all_alt_count)+sum(all_asis_turns))
    print('Average assistant turns per dialouge: ', sum(all_asis_turns)/(ctr+1))
    print('Average alt tags per dialouge: ', sum(all_alt_count)/(ctr+1))  
    print('Maximum Assistant Turns: ', max(all_asis_turns))
    print('Minimum Assistant Turns: ', min(all_asis_turns))



def main():
    data_path = 'socratic-debugging-benchmark/socratic_debugging_benchmark/v2_sigcse'
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'testset')

    # train stats
    print('##### Train Stats #####')
    retrive_stats(train_path)

    # test stats
    print('##### Test Stats #####')
    retrive_stats(test_path)

if __name__ == '__main__':
    main()