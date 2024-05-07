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

Some example commands include: 

1. Standard Fine-Tuning 

```
python finetune/sft_dpo.py --sft --base_model codellama/CodeLlama-7b-Instruct-hf --model_name codellama_sft_b2 --batch_size 2 --grad_accum_steps 32 --epochs 5
```

2. DPO
```
python finetune/sft_dpo.py --dpo --base_model codellama/CodeLlama-7b-Instruct-hf --model_name codellama_sft_b2 --pt_model_name codellama_sft_b2 --batch_size 1 --grad_accum_steps 64 --epochs 2
```

3. Generate 
```
python finetune/sft_dpo.py --generate --model_name codellama_sft_b2_dpo --pt_model_name codellama_sft_b2 --decoding greedy
```

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