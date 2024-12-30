import torch
import os
import torch
import json
import glob
import os
import datasets
import re

import random

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

random.seed(666)
num_feature = 1000

import pickle


def get_sample_data(data_path : dict  = {"code":"./data/code_alpaca_20k.json", "math":"./data/MetaMathQA-395K.json", "IF": "./data/alpaca_gpt4_data.json", "knowledge": "./data/rc/", "tool": "./data/training-data"}):
    all_data = {"math":[], "code":[],"IF":[], "knowledge": [], "tool":[]}
    for ids, (name, data_p) in enumerate(data_path.items()):
        print(name)

        if name == "IF":
            data = read_alpaca_data(data_p)
        if name == "code":
            data = read_codealpaca_data(data_p)
        if name == "math":
            data = read_metamathqa_data(data_p)
        if name == "knowledge":
            data = read_triviaqa_data(data_p)
        if name == "tool":
            data = read_apibank_data(data_p)
            
        all_data[name] = data
    file = open('./all_sample_data.pickle', 'wb')
    pickle.dump(all_data, file)
    file.close()
    

def read_alpaca_data(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        sample_list = random.sample(range(1, len(json_data)), num_feature)
        json_data = [json_data[i] for i in sample_list]

        for data in json_data:
            cut_length = random.randint(1, len(data["output"]))
            datas.append("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + data["instruction"] + data["input"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + data["output"][:cut_length])
    
    return datas

def read_triviaqa_data(data_path):
    datas = []
    parquet_files = glob.glob(os.path.join(data_path, '*.parquet'))  
    files = [name for name in parquet_files if "train" in name]
    dataset = datasets.load_dataset(
        "parquet",
        data_files={'train':files}
    )
    sample_list = random.sample(range(1, len(dataset["train"])), num_feature)
    json_data = [dataset["train"][i] for i in sample_list]
    for data in json_data:
        cut_length = random.randint(1, len(data['answer']['value']))
        datas.append("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + data['question']  + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +  data['answer']['value'][:cut_length])
    
    return datas

def read_apibank_data(data_path):
    datas = []
    processed_data = process(data_path + '/lv1-train.json')
    with open('./apibank_training_chat_v1_template.json', 'w', encoding='utf-8') as fout:
        json.dump(processed_data, fout, ensure_ascii=False, indent=4)
    processed_data = process(data_path + '/lv2-train.json')
    with open('./apibank_training_chat_v2_template.json', 'w', encoding='utf-8') as fout:
        json.dump(processed_data, fout, ensure_ascii=False, indent=4)
    processed_data = process(data_path + '/lv3-train.json')
    with open('./apibank_training_chat_v3_template.json', 'w', encoding='utf-8') as fout:
        json.dump(processed_data, fout, ensure_ascii=False, indent=4)

    file_list = ['./apibank_training_chat_v1_template.json', './apibank_training_chat_v2_template.json', './apibank_training_chat_v3_template.json']  
    json_data = merge_json_files(file_list)
    sample_list = random.sample(range(1, len(json_data)), num_feature)
    json_data = [json_data[i] for i in sample_list]

    for data in json_data:
            cut_length = random.randint(1, len(data["output"]))
            datas.append(data["instruction"] + data["input"] + data["output"][:cut_length])
    
    return datas

def read_codealpaca_data(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        json_filter = []
        for data in json_data:
            if len(data['output']) > 0:
                json_filter.append(data)
        sample_list = random.sample(range(1, len(json_filter)), num_feature)
        json_data = [json_filter[i] for i in sample_list]

        for data in json_data:
            cut_length = random.randint(1, len(data["output"]))
            datas.append("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + data["instruction"] + data["input"]  + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +  data["output"][:cut_length])
    
    return datas

def read_metamathqa_data(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        sample_list = random.sample(range(1, len(json_data)), num_feature)
        json_data = [json_data[i] for i in sample_list]
        
        for data in json_data:
            cut_length = random.randint(1, len(data["response"]))
            datas.append("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + data["query"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + data["response"][:cut_length])
    
    return datas


def parse_dialogue(text):
    # recognize the speaker in the conversation
    pattern = re.compile(r'(User|API-Request|AI):\s*(.*?)\s*(?=User:|API-Request:|AI:|$)', re.DOTALL)
    matches = pattern.findall(text)
    pro_dialogues = []
    dialogues = [(match[0], match[1].strip()) for match in matches]
    for dia in dialogues:
        if dia[0] == "API-Request":
            pattern_api = re.compile(r'(\s*.*?\s*)->(\s*.*?\s*)(?=$)', re.DOTALL)
            matches_api = pattern_api.findall(dia[1])
            assert len(matches_api) == 1
            assert len(matches_api[0])==2
            pro_dialogues.append(['assistant', "API-Request: " + matches_api[0][0]])
            pro_dialogues.append(['observation', matches_api[0][1]])
        else:
            dia_rename = None
            if dia[0] == 'User':
                dia_rename = ['user', dia[1]]
            elif dia[0] == 'AI':
                dia_rename = ['assistant', dia[1]]
            else:
                raise NameError
            pro_dialogues.append(dia_rename)

    # if the two dialogues both from assistant, merge them
    prefix = None
    merge = 0
    for i in range(len(pro_dialogues)-1):
        if pro_dialogues[i][0] == "assistant" and pro_dialogues[i+1][0] == "assistant":
            merge=merge+1
            print("---------------------------------------------------------")
            print(pro_dialogues)
            print(len(pro_dialogues)-1)
            print("---------------------------------------------------------")
            print("merged dialogues")
            print(pro_dialogues[i][1], pro_dialogues[i+1][1])
            print("---------------------------------------------------------")
            pro_dialogues[i][1] = pro_dialogues[i][1] + pro_dialogues[i+1][1]
            del pro_dialogues[i+1]
        if merge:
            print(merge, i, len(pro_dialogues)-1)
    if pro_dialogues[-1][0] == "assistant":
        prefix = pro_dialogues[-1][1]
        pro_dialogues = pro_dialogues[:-1]
    return pro_dialogues, prefix


def process(path):
    print("*"*20)
    print("current path", path)
    processed_data = []
    num = 0
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            input = data["input"]
            output = data["output"]
            if "AI:" == output[:3]:
                output = output[4:]
            split_user = input.split("User:")
            api_description = split_user[0]
            if "Generate AI Response: " in input:
                input = input[:-len("Generate AI Response: ")]
            if "Generate API Request: " in input:
                input = input[:-len("Generate API Request: ")]
            dialogues, prefix = parse_dialogue(input[len(api_description):])
            output = prefix + output if prefix != None else output
            try:
                assert len(dialogues) % 2 != 0
                for i in range(0, len(dialogues), 2):
                    assert dialogues[i][0] == 'user' or dialogues[i][0] == 'observation'
                    if i+1 < len(dialogues):
                        assert dialogues[i+1][0] == 'assistant'
            except:
                num += 1
                continue
            default_system = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
                    "Additional, you can utilize external tools to help handle the user's query if needed. "
                    "For example, you can generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests. The correct format is: 'API-Request: [ApiName(key1='value1', key2='value2', ...)]'. "
                    "And you can also generate a response based on the user's utterance and API Requests. "
                    )
            all_connect_input = ""
            
            all_connect_input = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + default_system + "API descriptions: " + api_description + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + dialogues[0][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            for i in range(1, len(dialogues) + 1, 2):
                if i == len(dialogues):
                    all_connect_input += prefix if prefix != None else ""
                else:
                    all_connect_input += dialogues[i][1] + "<|eot_id|>"
                    if dialogues[i+1][0] == "user": 
                        all_connect_input += "<|start_header_id|>user<|end_header_id|>\n\n" + dialogues[i+1][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    elif dialogues[i+1][0] == "observation":
                        all_connect_input += "<|start_header_id|>tool<|end_header_id|>\n\n" + dialogues[i+1][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    else:
                        raise ValueError
                    
            processed = {}
            processed['instruction'] = ""
            processed['input'] = all_connect_input
            processed['output'] = output
            processed_data.append(processed)

        print("the filtered data percent: ", num/len(json_data))
    return processed_data



    

def merge_json_files(file_list):
    merged_data = []
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            merged_data.extend(json_data)
    return merged_data


    

if __name__ == "__main__":
    get_sample_data()
    