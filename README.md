# Socratic Question Generation

## Requirements 

This codebase requires ```pytorch``` and ```huggingface``` to be installed which can be installed using ```pip```.

## Generate Invalid questions

```
python generate_bad_questions.py
python classifiy_gen_questions.py
python filter_bad_questions.py
python create_preference_dataset.py
```

The invalid questions are stored in ```bad_questions.json```. The classification results are stored in ```classifiy_gen_questions.json```. The correct invalid questions are stored in ```valid_bad_questions.json```. The preference dataset is stored in ```create_preference_dataset.py```.

## Preference Optimization 

```
python finetune/sft_dpo.py
```

This code takes several arguments which can be seen using the ```-h``` flag. ```--sft``` flag corresponds to standard fine-tuning and ```--dpo``` corresponds to direct preference optimization.

## LLama Zero-Shot Experiments 

```
python zero-shot_llama_prompt.py 
```

This code takes an argument ```--prompt cot``` for chain-of-thought prompting.

## Evaluate 

```
python evaluate.py --result_file <path_to_results_file>
```

Additionally, the code takes two arguments ```--zero True``` and ```--cot True``` for LLama zero-shot and CoT respectively. 