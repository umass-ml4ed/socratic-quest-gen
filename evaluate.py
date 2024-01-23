'''
script for evaluating the results based on the original paper
'''

import json
import os
import sys
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
import pandas as pd
import torch
import networkx as nx
from networkx.algorithms.matching import max_weight_matching
# import for metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

def compute_bert_score(predictions, references):
    '''
    returns the BERTScore (P, R, F1) between the predicted and gt string
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P, R, F1 = score(predictions, references, lang="en", model_type="microsoft/deberta-xlarge-mnli", device=device)
    return F1.mean().item()

def compute_rouge_score(predictions, references):
    '''
    returns the f1 rouge score between the predicted and gt string
    '''
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(predictions[0], references[0])
    return scores['rougeL'].fmeasure

def compute_bleu_score(predictions, references):
    '''
    returns the BLEU-4 score between the predicted and gt string
    '''
    reference = [references[0].split()]  # BLEU score function expects tokenized sentences
    prediction = predictions[0].split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference, prediction, weights=(1, 0, 0, 0), smoothing_function=smoothie)  # weights for BLEU-4
    return score

def compute_thoroughness(predictions, references, log=False, mode='rouge'):
    assert len(predictions) == len(references), "Length of predictions and references must be the same."
    
    # handle if both lists are empty
    if len(predictions) == 0 and len(references) == 0:
        return {
            'true_positives': [],
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    # ensure that predictions and references are lists of lists
    assert isinstance(predictions[0], list) and isinstance(references[0], list), "predictions and references must be lists of lists"

    true_positives = []
    precisions = []
    recalls = []
    f1s = []
    # if log use tqdm to show progress bar else use zip make it concise code-wise

    if log:
        lambda_func = tqdm(zip(predictions, references), total=len(predictions), desc="Computing thoroughness")
    else:
        lambda_func = zip(predictions, references)
    
    for pred_list, ref_list in lambda_func:
        # Create a bipartite graph
        B = nx.Graph()
        for i, p in enumerate(pred_list):
            B.add_node(f"0_{i}_{p}", bipartite=0)
        for i, r in enumerate(ref_list):
            B.add_node(f"1_{i}_{r}", bipartite=1)

        # Add edges with weight as the score between predictions and references
        for i, pred in enumerate(pred_list):
            for j, ref in enumerate(ref_list):
                # TODO: Replace with appropriate score 
                if mode == 'rouge':
                    score = compute_rouge_score(predictions=[pred], references=[ref])
                elif mode == 'bleu':
                    score = compute_bleu_score(predictions=[pred], references=[ref])
                elif mode == 'bert':
                    score = compute_bert_score(predictions=[pred], references=[ref])
                # use self._score_key(score) to get the score since score is a dict
                pred_id = f"0_{i}_{pred}"
                ref_id = f"1_{j}_{ref}"
                B.add_edge(pred_id, ref_id, weight=score)

        # Find maximum bipartite matching using the weight as the score
        matching = max_weight_matching(B)
        # change matching to a dict of pred: ref
        matching_dict = {}
        for pred, ref in matching:
            if pred.startswith('0_'):
                matching_dict[pred] = ref
            elif ref.startswith('0_'):
                matching_dict[ref] = pred
        # Compute the total score of the maximum matching
        tp = sum(B[pred][ref]['weight'] for pred, ref in matching_dict.items() if pred.startswith('0_'))
        true_positives.append(tp)
        precision = tp / len(pred_list)
        precisions.append(precision)
        recall = tp / len(ref_list)
        recalls.append(recall)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)

    return {
        'true_positives': true_positives,
        'precision': sum(precisions) / len(precisions),
        'recall': sum(recalls) / len(recalls),
        'f1': sum(f1s) / len(f1s)
    }


def process_pred(pred_str):
    '''
    extracts the predictied socratic guidance 
    '''
    # convert to list
    pred_str_list = literal_eval(pred_str)
    clean_pred_str_list = []
    for pred_str in pred_str_list:
        # remove </CONVERSATION>
        pred_str = pred_str.split('</CONVERSATION>')[0]
        # remove all </s>
        pred_str = pred_str.replace('</s>', '')
        # add to clean list
        clean_pred_str_list.append(pred_str)

    return clean_pred_str_list

def main():

    result_file = 'results/qg_results_codellama_sft_b2_greedy.csv'
    df = pd.read_csv(result_file)

    gt_outputs = [] # list of list of str
    pred_outputs = [] # list of list of str

    for i, row in df.iterrows():
        gt_output_str = row['output']
        gt_out_lst = literal_eval(gt_output_str)
        pred_str_lst = process_pred(row['prediction'])
        # append data to lists
        gt_outputs.append(gt_out_lst)
        pred_outputs.append(pred_str_lst)
    

    # print('Sample Outputs')
    # print(gt_outputs[0])
    # print(pred_outputs[0])

    all_metric_results = dict()

    # compute output scores
    # 1. Rouge
    metric_result_dict = compute_thoroughness(pred_outputs, gt_outputs, log=False)
    print('### Rouge ###')
    print('Precision: ', metric_result_dict['precision'])
    print('Recall: ', metric_result_dict['recall'])
    print('F1: ', metric_result_dict['f1'])
    all_metric_results['rouge'] = {'precision': metric_result_dict['precision'], 'recall': metric_result_dict['recall'], 'f1': metric_result_dict['f1']}

    # 2. BLEU
    print('\n\n### BLEU ###')
    metric_result_dict = compute_thoroughness(pred_outputs, gt_outputs, log=False, mode='bleu')
    print('Precision: ', metric_result_dict['precision'])
    print('Recall: ', metric_result_dict['recall'])
    print('F1: ', metric_result_dict['f1'])
    all_metric_results['bleu'] = {'precision': metric_result_dict['precision'], 'recall': metric_result_dict['recall'], 'f1': metric_result_dict['f1']}

    # 3. BERT
    print('\n\n### BERT ###')
    metric_result_dict = compute_thoroughness(pred_outputs, gt_outputs, log=True, mode='bert')
    print('Precision: ', metric_result_dict['precision'])
    print('Recall: ', metric_result_dict['recall'])
    print('F1: ', metric_result_dict['f1'])
    all_metric_results['bert'] = {'precision': metric_result_dict['precision'], 'recall': metric_result_dict['recall'], 'f1': metric_result_dict['f1']}

    # store metric results for all three metrics in csv file
    metric_df = pd.DataFrame(all_metric_results).T
    save_dir = 'metrics_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results_file_name = result_file.split('/')[-1].split('.')[0]
    metric_df.to_csv(os.path.join(save_dir, f'{results_file_name}.csv'))


if __name__ == '__main__':
    main()
