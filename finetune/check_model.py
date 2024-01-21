from sft_dpo import *

def check_model(model, tokenizer):
    '''
    check the working of the loaded omdel for text completion
    '''
    input_text = "```python\nimport numpy as np"
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    # Generate text
    output = model.generate(input_ids=input_ids, max_length=100, temperature=0.7, do_sample=True)
    # Decode the output
    completed_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(completed_text)

def main():
    model, tokenizer = get_model('codellama/CodeLlama-7b-Instruct-hf', None, None, False, False)

if __name__ == '__main__':
    main()
