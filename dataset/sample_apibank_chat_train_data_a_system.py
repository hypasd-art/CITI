import json
import random
import glob
import os
import datasets
random.seed(1)

default_system = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
                    "Additional, you can utilize external tools to help handle the user's query if needed. "
                    "For example, you can generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests. The correct format is: 'API-Request: [ApiName(key1='value1', key2='value2', ...)]'. "
                    "And you can also generate a response based on the user's utterance and API Requests. "
                    )

def reading_metamathqa(path):
    data_processed = []
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            data_processed.append({"system": default_system, "original_query":"", "tools":"", "input": ["user", data["query"]], "output": data["response"], "history": [], "task_types": 0})
        return data_processed

def reading_codealpaca(path):
    data_processed = []
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            data_processed.append({"system": default_system, "original_query":"", "tools":"", "input": ["user", data["instruction"] + data["input"]], "output": data["output"], "history": [], "task_types": 0})
        
    return data_processed

def reading_triviaqa(path):
    data_processed = []
    parquet_files = glob.glob(os.path.join(path, '*.parquet'))  
    files = [name for name in parquet_files if "train" in name]
    dataset = datasets.load_dataset(
        "parquet",
        data_files={'train':files}
    )
    for data in dataset["train"]:
        data_processed.append({"system": default_system, "original_query":"", "tools":"", "input":['user', data['question']], "output": data['answer']['value'], "history": [], "task_types": 0})
    
    return data_processed

def reading_alpaca_gpt4(path):
    data_processed = []
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            data_processed.append({"system": default_system, "original_query":"", "tools":"", "input": ["user", data["instruction"] + data["input"]], "output": data["output"], "history": [], "task_types": 0})
        
    return data_processed


merged_data = []
data_metamathqa = reading_metamathqa('./MetaMathQA/MetaMathQA-395K.json')
data_codealpaca = reading_codealpaca('./CodeAlpaca-20k/code_alpaca_20k.json')
data_triviaqa = reading_triviaqa('./trivia_qa/rc')
data_alpaca_gpt4 = reading_alpaca_gpt4('./GPT-4-LLM/data/alpaca_gpt4_data.json')
with open('./API-Bank/apibank_training_chat_a_system.json', 'r', encoding='utf-8') as f:
    data_apibank = json.load(f)
    for data in data_apibank:
        data["task_types"] = 1

sample_list = random.sample(range(1, len(data_alpaca_gpt4)), 5000)
merged_data.extend([data_alpaca_gpt4[i] for i in sample_list])

sample_list = random.sample(range(1, len(data_metamathqa)), 5000)
merged_data.extend([data_metamathqa[i] for i in sample_list])

sample_list = random.sample(range(1, len(data_triviaqa)), 5000)
merged_data.extend([data_triviaqa[i] for i in sample_list])

sample_list = random.sample(range(1, len(data_codealpaca)), 5000)
merged_data.extend([data_codealpaca[i] for i in sample_list])

merged_data.extend(data_apibank)


random.shuffle(merged_data)
with open('./apibank_training_chat_a_system_merge.json', 'w', encoding='utf-8') as fout:
    json.dump(merged_data, fout, ensure_ascii=False, indent=4)
