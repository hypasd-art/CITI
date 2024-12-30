import json
import random
import re
random.seed(1)


instruction_dict = {
    "level-1-api.json": "Please generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the previous dialogue context.\nThe current year is 2023.\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\n",
    "level-1-response.json": "Please generate a response as an AI assistant based on the previous dialogue context and API request.\nThe current year is 2023. The format of API Calls is: 'API-Request: [ApiName(key1='value1', key2='value2', ...)]'\n\nExpected output:\nAI's response\n\n",
    "level-2-api.json": "Please generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the previous dialogue context.\nThe current year is 2023.\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\n",
    "level-2-response.json": "Please generate a response as an AI assistant based on the previous dialogue context and API request. The format of API Calls is: 'API-Request: [ApiName(key1='value1', key2='value2', ...)]'\nThe current year is 2023.\n\nExpected output:\nAI's response\n\n",
    "level-3-batch-inf-icl.json": "\nYou will be tested on your ability to make multiple API calls to fulfill a requirement based on a single sentence. You will be given an API box that includes a set of APIs such as a calculator, translator, WikiSearch, etc. When you want to use an API, you have to search for it in the API search engine using keywords. Try to describe it with these keywords. The tool search engine will then return you the most relevant information about the tool (api name, description, input/output parameters).\n\nAfter you give each API call, stop generating and wait for input, I will return the results of the API call to you and make the next call based on the results. Your output is only an API call and does not contain any explanatory text, which means starts with [ and ends with ].\nHere is an example of a test where ChatGPT represents you and API represents the return value.\nExample:\nRequirement: calculate the result of (5+3)*6 and sum the with 5.\nChatGPT: API-Request: [ToolSearcher(keywords='calculator')]\nAPI: {\"name\": \"Calculator\", \"description\": \"This API provides basic arithmetic operations: addition, subtraction, multiplication, and division.\", \"input_parameters\": {\"formula\": {\"type\": \"str\", \"description\": \"The formula that needs to be calculated. Only integers are supported. Valid operators are +, -, *, /, and (, ). For example, '(1 + 2) * 3'.\"}}, \"output_parameters\": {\"result\": {\"type\": \"float\", \"description\": \"The result of the formula.\"}}}\nChatGPT: API-Request: [Calculator(formula='(5+3)*6')]\nAPI: {'result': 48}\nChatGPT: API-Request: [Calculator(formula='48+5')]\nAPI: {'result': 53}\n\n",
    "level-3-batch-inf-response.json": "Please generate a response as an AI assistant based on the previous dialogue context and API request. The format of API Calls is: 'API-Request: [ApiName(key1='value1', key2='value2', ...)]'\nThe current year is 2023.\n\nExpected output:\nAI's response\n\n",
}


def parse_dialogue(text):
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
    processed_dialogues = []
    i = 0
    while True:
        if i == len(pro_dialogues):
            break
        if i != len(pro_dialogues) - 1 and pro_dialogues[i][0] == "assistant" and pro_dialogues[i+1][0] == "assistant":
            processed_dialogues.append(["assistant", pro_dialogues[i][1] + pro_dialogues[i+1][1]])
            del pro_dialogues[i+1]
        elif i != len(pro_dialogues) - 1 and pro_dialogues[i][0] == "observation" and pro_dialogues[i+1][0] == "user":
            processed_dialogues.append(pro_dialogues[i])
            processed_dialogues.append(["assistant", ""])
            processed_dialogues.append(pro_dialogues[i+1])
            del pro_dialogues[i+1]
        else:
            processed_dialogues.append(pro_dialogues[i])
        i += 1
    
    if processed_dialogues[-1][0] == "assistant":
        prefix = processed_dialogues[-1][1]
        processed_dialogues = processed_dialogues[:-1]
    return processed_dialogues, prefix


def process(path):
    print("*"*20)
    print("current path", path)
    processed_data = []
    num = 0
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            instruction = data["instruction"]
            input = data["input"]
            output = data["expected_output"]

            # api_description = "API descriptions: " + instruction.split("API descriptions:")[1]
            api_description = "API descriptions: " + instruction.split("API descriptions:")[1]
            instruction_new = instruction_dict[path.split("/")[-1]] + api_description

            
            if "\nGenerate AI Response:\n" in input:
                input = input[:-len("\nGenerate AI Response:\n")]
                prompt = "Generate AI Response:"
            if "\nGenerate API Request:\n" in input:
                input = input[:-len("\nGenerate API Request:\n")]
                prompt = "Generate API Request:"

            dialogues, prefix = parse_dialogue(input)
            ################################################
            output = output
            try:
                assert len(dialogues) % 2 != 0
                for i in range(0, len(dialogues), 2):
                    assert dialogues[i][0] == 'user' or dialogues[i][0] == 'observation'
                    if i+1 < len(dialogues):
                        assert dialogues[i+1][0] == 'assistant'
            except:
                print("\n\nunvalid data format")
                
                print("--------------------------input--------------------------")
                print(input)
                print("--------------------------output--------------------------")
                print(output)
                print("--------------------------dialogues--------------------------")
                for k in dialogues:
                    print(k)
                print(len(dialogues))
                num += 1
                continue
            all_connect_input = ""
            
            all_connect_input = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + instruction_new + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + dialogues[0][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            for i in range(1, len(dialogues) + 1, 2):
                if i == len(dialogues):
                    all_connect_input += prefix if prefix != None else ""
                    all_connect_input += "\n" + prompt
                else:
                    all_connect_input += dialogues[i][1] + "<|eot_id|>"
                    if dialogues[i+1][0] == "user": 
                        all_connect_input += "<|start_header_id|>user<|end_header_id|>\n\n" + dialogues[i+1][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    elif dialogues[i+1][0] == "observation":
                        all_connect_input += "<|start_header_id|>tool<|end_header_id|>\n\n" + dialogues[i+1][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    else:
                        raise ValueError
                    
            processed = {}
            processed['file'] = data['file']
            processed['id'] = data['id']
            processed['instruction'] = ""
            processed['input'] = all_connect_input
            processed['expected_output'] = output

            
            processed_data.append(processed)

        print("the filtered data percent: ", num/len(json_data))

    return processed_data


def process_level_3(path):
    print("*"*20)
    print("current path", path)

    processed_data = []
    num = 0
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            # instruction = data["instruction"]
            instruction = instruction_dict[path.split("/")[-1]]
            input = data["input"]
            output = data["output"]

            # api_description = "API descriptions: " + instruction.split("API descriptions:")[1]
            split_user = input.split("User:")
            api_description = split_user[0]

            if "\nGenerate AI Response: \n" in input:
                input = input[:-len("\nGenerate AI Response: \n")]
                prompt = "Generate AI Response:"
            if "\nGenerate API Request: \n" in input:
                input = input[:-len("\nGenerate API Request: \n")]
                prompt = "Generate API Request:"

            dialogues, prefix = parse_dialogue(input[len(api_description):])
            ################################################
            output = output
            try:
                assert len(dialogues) % 2 != 0
                for i in range(0, len(dialogues), 2):
                    assert dialogues[i][0] == 'user' or dialogues[i][0] == 'observation'
                    if i+1 < len(dialogues):
                        assert dialogues[i+1][0] == 'assistant'
            except:
                print("\n\nunvalid data format")
                
                print("--------------------------input--------------------------")
                print(input)
                print("--------------------------output--------------------------")
                print(output)
                print("--------------------------dialogues--------------------------")
                for k in dialogues:
                    print(k)
                print(len(dialogues))
                num += 1
                continue
            all_connect_input = ""
            all_connect_input = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + instruction + "API descriptions: " + api_description + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + dialogues[0][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            for i in range(1, len(dialogues) + 1, 2):
                if i == len(dialogues):
                    all_connect_input += prefix if prefix != None else ""
                    all_connect_input += "\n" + prompt
                else:
                    all_connect_input += dialogues[i][1] + "<|eot_id|>"
                    if dialogues[i+1][0] == "user": 
                        all_connect_input += "<|start_header_id|>user<|end_header_id|>\n\n" + dialogues[i+1][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    elif dialogues[i+1][0] == "observation":
                        all_connect_input += "<|start_header_id|>tool<|end_header_id|>\n\n" + dialogues[i+1][1] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    else:
                        raise ValueError
            

            processed = {}
            processed['sample_id'] = data['sample_id']
            processed['api_id'] = data['api_id']
            processed['instruction'] = ""
            processed['input'] = all_connect_input
            processed['output'] = output

            processed_data.append(processed)

        print("the filtered data percent: ", num/len(json_data))

    return processed_data

# for 
for test_dir in ['level-1-api', 'level-1-response', 'level-2-api', 'level-2-response']:
    data_dir = "./test-data/" + test_dir +".json"
    processed_data = process(data_dir)
    with open('./test-data-llama3-instruct/'+test_dir+".json", 'w', encoding='utf-8') as fout:
        json.dump(processed_data, fout, ensure_ascii=False, indent=2)
for test_dir in ['level-3-batch-inf-icl', 'level-3-batch-inf-response',]:
    data_dir = "./test-data/" + test_dir +".json"
    processed_data = process_level_3(data_dir)
    with open('./test-data-llama3-instruct/'+test_dir+".json", 'w', encoding='utf-8') as fout:
        json.dump(processed_data, fout, ensure_ascii=False, indent=2)


    
