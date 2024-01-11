'''
Classifies generated bad questions using an LLM
'''
from utils import *

def main():
    system_message, few_shot_prompt = load_classification_prompt()
    print('### System Message ###')
    print(system_message)
    print('### Few Shot Prompt ###')
    print(few_shot_prompt)

if __name__ == '__main__':
    main()