import json
import random
import re
random.seed(1)
file_list = ['./apibank_training_chat_v1_a_system.json', './apibank_training_chat_v2_a_system.json', './apibank_training_chat_v3_a_system.json']  

default_system = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
                    "Additional, you can utilize external tools to help handle the user's query if needed. "
                    "For example, you can generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests. The correct format is: 'API-Request: [ApiName(key1='value1', key2='value2', ...)]'. "
                    "And you can also generate a response based on the user's utterance and API Requests. "
                    )

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

    # print(pro_dialogues)
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

def is_prefix(list1, list2):
    if len(list1) > len(list2):
        return False
    
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            print("unprefix")
            print(list1[i])
            print(list2[i])
            print("\n")
            return False
    
    return True

def remove_duplicates(data):
    max_history_length = {}
    
    for item in data:
        key = (item['original_query'], item['tools'])
        history_length = len(item['history'])
        
        if key not in max_history_length:
            max_history_length[key] = history_length
        else:
            max_history_length[key] = max(max_history_length[key], history_length)
    
    result = []
    seen_keys = {}
    
    for item in data:
        key = (item['original_query'], item['tools'])
        if key not in seen_keys.keys() and len(item['history']) == max_history_length[key]:
            result.append(item)
            seen_keys[key] = item['history']

    for item in data:
        key = (item['original_query'], item['tools'])
        assert is_prefix(item["history"], seen_keys[key])
    return result

def process(path):
    print("*"*20)
    print("current path", path)
    processed_data = []
    num = 0
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            chat_input = None
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
            user_query = dialogues[0][1]
            try:
                assert len(dialogues) % 2 != 0
                for i in range(0, len(dialogues), 2):
                    assert dialogues[i][0] == 'user' or dialogues[i][0] == 'observation'
                    if i+1 < len(dialogues):
                        assert dialogues[i+1][0] == 'assistant'
            except:
                num += 1
                continue
            history = []
            for i in range(0, len(dialogues), 2):
                if i+1 == len(dialogues):
                    chat_input = dialogues[i]
                else:
                    history.append([dialogues[i], dialogues[i+1]])
            processed = {}
            processed['system'] = default_system
            processed['original_query'] = user_query
            processed['tools'] = "API descriptions:\n" + api_description
            processed['input'] = chat_input
            processed['output'] = output
            processed['history'] = history
            assert chat_input != None
            processed_data.append(processed)

        print("the filtered data percent: ", num/len(json_data))
    return processed_data

processed_data = process('./training-data/lv1-train.json')
with open('apibank_training_chat_v1_a_system.json', 'w', encoding='utf-8') as fout:
    json.dump(processed_data, fout, ensure_ascii=False, indent=4)
processed_data = process('./training-data/lv2-train.json')
with open('apibank_training_chat_v2_a_system.json', 'w', encoding='utf-8') as fout:
    json.dump(processed_data, fout, ensure_ascii=False, indent=4)
processed_data = process('./training-data/lv3-train.json')
with open('apibank_training_chat_v3_a_system.json', 'w', encoding='utf-8') as fout:
    json.dump(processed_data, fout, ensure_ascii=False, indent=4)

    

def merge_json_files(file_list):
    merged_data = []
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            merged_data.extend(json_data)
    return merged_data

merged_data = merge_json_files(file_list)
random.shuffle(merged_data)
with open('apibank_training_chat_a_system.json', 'w', encoding='utf-8') as fout:
    json.dump(merged_data, fout, ensure_ascii=False, indent=4)
