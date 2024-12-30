import json
import random
import glob
import os
import datasets
random.seed(1)



def reading_metamathqa_llama3(path):
    data_processed = []
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            data_processed.append({"instruction":"", "input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + data["query"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "output": data["response"]+ "<|eot_id|>"})
        return data_processed
    
p_data3 = reading_metamathqa_llama3("/mnt/userdata/yphao/CITI/lm_evaluation_harness/lm_eval/datasets/metamathqa/MetaMathQA-395K.json")
with open("/mnt/userdata/yphao/CITI/lm_evaluation_harness/lm_eval/datasets/metamathqa/MetaMathQA-395K-llama3.json", 'w') as fb:
    json.dump(p_data3, fb, indent=2)


def reading_metamathqa_phi3(path):
    data_processed = []
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            data_processed.append({"instruction":"", "input": "<s><|user|>\n" + data["query"] + "<|end|>\n<|assistant|>\n", "output": data["response"]+ "<|end|>"})
        return data_processed
    
p_data3 = reading_metamathqa_phi3("/mnt/userdata/yphao/CITI/lm_evaluation_harness/lm_eval/datasets/metamathqa/MetaMathQA-395K.json")
with open("/mnt/userdata/yphao/CITI/lm_evaluation_harness/lm_eval/datasets/metamathqa/MetaMathQA-395K-phi3.json", 'w') as fb:
    json.dump(p_data3, fb, indent=2)

def reading_metamathqa_mistral(path):
    data_processed = []
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            data_processed.append({"instruction":"", "input": "<s>[INST] " + data["query"] + " [/INST]", "output": data["response"]+ "</s>"})
        return data_processed
    
p_data3 = reading_metamathqa_mistral("/mnt/userdata/yphao/CITI/lm_evaluation_harness/lm_eval/datasets/metamathqa/MetaMathQA-395K.json")
with open("/mnt/userdata/yphao/CITI/lm_evaluation_harness/lm_eval/datasets/metamathqa/MetaMathQA-395K-mistral.json", 'w') as fb:
    json.dump(p_data3, fb, indent=2)