import json

def save_dict_to_file(dictionary, filename='data.json'):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(dictionary, file, ensure_ascii=False, indent=4)
        print(f"saved moelora allocation to {filename}")
    except IOError as e:
        print(f"can't save: {e}")


def load_dict_from_file(filename='data.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            dictionary = json.load(file)
        print(f"loading moelora allocation from {filename}")
        return dictionary
    except IOError as e:
        print(f"can't read {e}")

important_score = {}
important_score_all = {}
important_score_all_w_tool = {}

important_score_align = {}
important_score_math = {}
important_score_code = {}
important_score_triviaqa = {}
important_score_tool = {}


componentdict = {0:"q_proj", 1:"k_proj", 2:"v_proj", 3:"o_proj", 4:"gate_proj", 5:"up_proj", 6:"down_proj"}

import torch
import pickle
head_importance_path = "/mnt/userdata/yphao/CITI/lm_evaluation_harness/logs/head_importance/llama3_chat/0shot_alpacallama3.pkl"
with open(head_importance_path, 'rb') as f:
    importance = pickle.load(f)
    importance_score_q_proj,importance_score_k_proj,importance_score_v_proj,importance_score_o_proj,gate_importance_score,up_importance_score,down_importance_score = importance[0], importance[1], importance[2], importance[3], importance[4], importance[5], importance[6]
    for i in range(len(importance)):
        for j in range(len(importance_score_k_proj)):
            if i >= 4:
                important_score_align["layers."+str(j)+".mlp."+componentdict[i]] = importance[i][j].item()
            else:
                important_score_align["layers."+str(j)+".self_attn."+componentdict[i]] = importance[i][j].item()

head_importance_path = "/mnt/userdata/yphao/CITI/lm_evaluation_harness/logs/head_importance/llama3_chat/0shot_metamathqallama3.pkl"
with open(head_importance_path, 'rb') as f:
    importance = pickle.load(f)
    importance_score_q_proj,importance_score_k_proj,importance_score_v_proj,importance_score_o_proj,gate_importance_score,up_importance_score,down_importance_score = importance[0], importance[1], importance[2], importance[3], importance[4], importance[5], importance[6]
    for i in range(len(importance)):
        for j in range(len(importance_score_k_proj)):
            if i >= 4:
                important_score_math["layers."+str(j)+".mlp."+componentdict[i]] = importance[i][j].item()
            else:
                important_score_math["layers."+str(j)+".self_attn."+componentdict[i]] = importance[i][j].item()

head_importance_path = "/mnt/userdata/yphao/CITI/lm_evaluation_harness/logs/head_importance/llama3_chat/0shot_codealpacallama3.pkl"
with open(head_importance_path, 'rb') as f:
    importance = pickle.load(f)
    importance_score_q_proj,importance_score_k_proj,importance_score_v_proj,importance_score_o_proj,gate_importance_score,up_importance_score,down_importance_score = importance[0], importance[1], importance[2], importance[3], importance[4], importance[5], importance[6]
    for i in range(len(importance)):
        for j in range(len(importance_score_k_proj)):
            if i >= 4:
                important_score_code["layers."+str(j)+".mlp."+componentdict[i]] = importance[i][j].item()
            else:
                important_score_code["layers."+str(j)+".self_attn."+componentdict[i]] = importance[i][j].item()

head_importance_path = "/mnt/userdata/yphao/CITI/lm_evaluation_harness/logs/head_importance/llama3_chat/0shot_triviaqallama3.pkl"
with open(head_importance_path, 'rb') as f:
    importance = pickle.load(f)
    importance_score_q_proj,importance_score_k_proj,importance_score_v_proj,importance_score_o_proj,gate_importance_score,up_importance_score,down_importance_score = importance[0], importance[1], importance[2], importance[3], importance[4], importance[5], importance[6]
    for i in range(len(importance)):
        for j in range(len(importance_score_k_proj)):
            if i >= 4:
                important_score_triviaqa["layers."+str(j)+".mlp."+componentdict[i]] = importance[i][j].item()
            else:
                important_score_triviaqa["layers."+str(j)+".self_attn."+componentdict[i]] = importance[i][j].item()

head_importance_path = "/mnt/userdata/yphao/CITI/lm_evaluation_harness/logs/head_importance/llama3_chat/0shot_toolalpacallama3.pkl"
with open(head_importance_path, 'rb') as f:
    importance = pickle.load(f)
    importance_score_q_proj,importance_score_k_proj,importance_score_v_proj,importance_score_o_proj,gate_importance_score,up_importance_score,down_importance_score = importance[0], importance[1], importance[2], importance[3], importance[4], importance[5], importance[6]
    for i in range(len(importance)):
        for j in range(len(importance_score_k_proj)):
            if i >= 4:
                important_score_tool["layers."+str(j)+".mlp."+componentdict[i]] = importance[i][j].item()
            else:
                important_score_tool["layers."+str(j)+".self_attn."+componentdict[i]] = importance[i][j].item()
                
from collections import OrderedDict  
  
def div_mean(important_data):
    mean = 0
    important_data2 = {}
    for v in important_data.values():
        mean += v
    for k, v in important_data.items():
        important_data2[k] = v / mean
    return important_data2

important_score_align =  div_mean(important_score_align)
important_score_math =  div_mean(important_score_math)
important_score_code =  div_mean(important_score_code)
important_score_triviaqa =  div_mean(important_score_triviaqa)
important_score_tool =  div_mean(important_score_tool)


for k,v in important_score_math.items():
    important_score_all[k] = (important_score_align[k] + important_score_math[k] + important_score_code[k] + important_score_triviaqa[k]) # + 2 * important_score_tool[k] 
sorted_dict_all = OrderedDict(sorted(important_score_all.items(), key=lambda x: x[1], reverse=False))  # 升序排序


for k,v in important_score_math.items():
    important_score_all_w_tool[k] = (important_score_align[k] + important_score_math[k] + important_score_code[k] + important_score_triviaqa[k])/4 + important_score_tool[k] 
sorted_dict_all_w_tool = OrderedDict(sorted(important_score_all_w_tool.items(), key=lambda x: x[1], reverse=False))  # 升序排序

sorted_dict_tool = OrderedDict(sorted(important_score_tool.items(), key=lambda x: x[1], reverse=False)) # 升序排序

import random
random.seed(1)




# base on the tool importance 
assert len(list(sorted_dict_tool.keys())[int(len(sorted_dict_tool) * 0.8):]) == 45
save_dict_to_file(list(sorted_dict_tool.keys())[int(len(sorted_dict_tool) * 0.8):], "/mnt/publiccache/yphao/model/toolalpaca_moelora_allocation/llama3_moe_allocation_top_20_percent.json")

print("---------------------------top 20% significant components on tool learning-----------------------------")
print(",".join(list(sorted_dict_tool.keys())[int(len(sorted_dict_tool) * 0.8):]))
print(len(list(sorted_dict_tool.keys())[int(len(sorted_dict_tool) * 0.8):]))
print("\n")




# based on the general task importance
save_dict_to_file(list(sorted_dict_all.keys())[:int(len(sorted_dict_all) * 0.1)], "/mnt/publiccache/yphao/model/toolalpaca_moelora_allocation/llama3_unfreeze_down_10.json")


