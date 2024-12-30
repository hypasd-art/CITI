import logging
import requests
from datetime import datetime
import torch
import gc
from utils import load_openapi_spec, escape
# from agent.tools import Tool, GetDetailsTool
from agent.agent_prompts import test_prompt_v1
from agent.custom_agent import CustomZeroShotAgent
import os
import json
import logging
import argparse

# import requests
from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# from agent.get_agent import get_agent
# from agent.agent_prompts import prompt_proj
from utils import load_openapi_spec, analyze_openapi_spec
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
from agent.tools import Tool, GetDetailsTool, tool_projection
# logger = logging.getLogger(__name__)
import time

def convert_type(ori_type):
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }

    type_mapping.update({j: i for i, j in type_mapping.items()})
    
    return type_mapping.get(ori_type)


def type_check(param_name, param_schema, input_params):
    type_check_error = []
    doc_type = convert_type(param_schema.get("type", ""))
    
    if doc_type and not isinstance(input_params[param_name], doc_type):
        type_error = True
        if (doc_type in [int, float, bool] and type(input_params[param_name]) == str) or \
            (doc_type == float and type(input_params[param_name]) == int):
            try:
                input_params[param_name] = doc_type(input_params[param_name])
                type_error = False
            except ValueError:
                pass
        if type_error:
            type_check_error.append((
                param_name,
                convert_type(doc_type),
                convert_type(type(input_params[param_name]))
            ))
    if "enum" in param_schema and input_params[param_name] != "" and\
        input_params[param_name] not in param_schema["enum"]:

        type_check_error.append((
            param_name,
            f'one of {param_schema["enum"]}',
            f'"{input_params[param_name]}"'
        ))
    return type_check_error
        

    
def call_api_function(input_params, openapi_spec, path, method, base_url=None):
    # print(openapi_spec)
    function_doc = openapi_spec["paths"][path][method]

    required_params = set()
    params = {
        "query": {},
        "header": {},
        "path": {},
        "cookie": {}
    }

    type_check_error = []
    for param_doc in function_doc.get("parameters", []):
        if param_doc.get("required"):
            required_params.add((param_doc["in"], param_doc["name"]))

        if param_doc["name"] in input_params:
            required_params.discard((param_doc["in"], param_doc["name"]))
            params[param_doc["in"]][param_doc["name"]] = input_params[param_doc["name"]]
            type_check_error.extend(type_check(param_doc["name"], param_doc, input_params))
    # print(type_check_error)
    body_data = None
    required_body_params = None
    # not in real
    if "requestBody" in function_doc:
        body_data = {}
        request_body_schema = function_doc["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})

        if "properties" in request_body_schema:
            required_body_params = set(request_body_schema.get("required", []))
            for property_name, property_value in request_body_schema["properties"].items():
                if property_name in input_params:
                    body_data[property_name] = input_params[property_name]
                    required_body_params.discard(property_name)
                    type_check_error.extend(type_check(property_name, property_value, input_params))
                    
    if len(type_check_error) > 0:
        error_str = "\n".join([f"Parameter type error: \"{i[0]}\", expected {i[1]}, but got {i[2]}. You need to change the input and try again." for i in type_check_error])
        raise ValueError(error_str)
    
    if required_params or required_body_params:
        missing_params = ", ".join([f'"{param[1]}"' for param in required_params])
        missing_params += [f'"{param}"' for param in required_body_params]
        raise ValueError(f"Missing required parameters: {', '.join(required_body_params)}. You need to change the input and try again.")

    base_url = openapi_spec['servers'][0]['url'] if base_url is None else base_url
    url = f"{base_url.rstrip('/')}{path.format(**params['path'])}"
    headers = {"Content-Type": "application/json"}
    headers.update(params["header"])

    logging.info("request url: {url}")
    logging.info(f"\t{method}\n\t{url}\n\t{params}\n\t{body_data}\n\t{headers}\t\n")

    # print(f"\t{method}\n\t{url}\n\t{params}\n\t{body_data}\n\t{headers}\t\n")
    response = requests.request(
        method=method.upper(),
        url=url,
        params=params["query"],
        json=body_data,
        headers=headers,
        cookies=params["cookie"]
    )

    if "image" in response.headers.get("Content-Type", ""):
        image_extension = response.headers.get("Content-Type", "").split("/")[-1]
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        image_path = f"./images/{timestamp}.{image_extension}"
        with open(image_path, 'wb') as f:
            f.write(response.content)
        response._content = bytes(f"Recieved an image, saved in '{image_path}'.", "utf-8")
    
    logging.debug("url: {response.request.url}")
    logging.debug("body: {response.request.body}")
    logging.debug("headers: {response.request.headers}")

    return response
import time
def func(params, openapi_spec, path, method, retrieval_available=False, base_url=None, max_observation_length=3000, **kwargs):
    """
    params: the action input
    openapi_spec: document store the information about function
    path: the first item in Function_Projection, e.g. "/api/v3/CountryInfo/{countryCode}"
    method: the second item in Function_Projection e.g. get
    """
    try:
        json_start = min([idx for idx in (params.find('{'), params.find('[')) 
                        if idx != -1])
        json_end = max([idx for idx in (params.rfind('}'), params.rfind(']')) 
                        if idx != -1]) + 1

        valid_json_string = params[json_start:json_end]
        params = json.loads(valid_json_string)
        # params = parse_json_string(params)
        if isinstance(params, list):
            return "'Action Input' cannot be a list. Only call one function per action."
    except:
        return "Invalid JSON format. Please ensure 'Action Input' is a valid JSON object."
    
    if path == "":
        return "You should generate function call or return the 'Final Answer' and explain what happened"
    retry_times = 0
    retry_times2 = 0

    response = None
    while retry_times < 5 and retry_times2 < 10:
        try:
            # print(params, path, method)
            response = call_api_function(
                input_params=params,
                openapi_spec=openapi_spec,
                path=path,
                method=method,
                # base_url=base_url
            )
        except ValueError as e:
            return str(e)
        except:
            retry_times2 += 1
            time.sleep(1)
            continue
        message = f"Status Code: {response.status_code}. Response: {response.text}"
        if response.status_code >= 500:
            retry_times += 1
            continue
        break
    if response == None:
        message = "Can not connect to the API due to network issues. You should choose one of: (1) change the input and retry; (2) return the 'Final Answer' and explain what happened; (You must choose this one when the error occurs more than 3 times.) (3) call another function."
        return message
    if not 200 <= response.status_code < 300:
        # message += ". You can try to change the input or call another function. "
        message += ". You should choose one of: (1) change the input and retry; (2) return the 'Final Answer' and explain what happened; (You must choose this one when the error occurs more than 3 times.) (3) call another function."
        
    max_output_len = max_observation_length
    if len(message) > max_output_len:
        if retrieval_available:
            file_name = f"./tmp/retrieval_{int(time.time())}.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(message)
            return "The output is too long. You need to use the 'retrievalDataFromFile' function to retrieve the output from the file: " + file_name
        else:
            message = message[:max_output_len]
    return message


def func_retrieval(params):
    from string import Template
    from utils import openai_chat_completions
    logging.info("retrieval data fromfile:\n")
    logging.info(params)
    logging.info("-------------------------------------------------------")
    try:
        json_start = min([idx for idx in (params.find('{'), params.find('[')) 
                        if idx != -1])
        json_end = max([idx for idx in (params.rfind('}'), params.rfind(']')) 
                        if idx != -1]) + 1

        valid_json_string = params[json_start:json_end]
        params = json.loads(valid_json_string)
        # params = parse_json_string(params)
        file_path = params["file_path"]
        query = params["query"]
    except:
        return "Invalid JSON format. Please ensure 'Action Input' is a valid JSON object."
    
    if not os.path.exists(file_path):
        return "File does not exist."
    
    file_content = open(file_path, "r", encoding="utf-8").read()
    template = Template(open("prompts/retrieval.txt", "r").read())
    prompt = template.substitute(output=file_content[:40000], query=query)
    completion = openai_chat_completions([{"role": "user", "content": prompt}], model="gpt-3.5-turbo-0125")
    outputs = completion["choices"][0]["message"]["content"]

    return {"retrieved_info": outputs}

def parse_output(output, openapi_spec, function_proj,retrieval_available, base_url):
    """
    return paras:
    observation,
    final_answer_or_not
    steps_information
    """
    if "Response:" in output:
        try:
            assert "Action:" not in output[output.find("Response:"):]
            assert "Action Input:" not in output[output.find("Response:"):]
        except:
            logging.info("Invalid output. Please do not call tools after give the final response")
            # print("Invalid output. Please do not call tools after give the final response") #'Action Input' "
        Final_thought = output.split("Response:")[0].split("Thought")[1]
        model_output = output.split("Response")[1]
        return "", True, [Final_thought, model_output]
    try:
        thought = ""
        action = ""
        action_input = ""
        path = ""
        method = ""
        thought = output.split("Action:")[0]
        action = output.split("Action:")[1].split("Action Input:")[0].strip()
        action_input = output.split("Action:")[1].split("Action Input:")[1].strip()
        path = function_proj[action][0]
        method = function_proj[action][1]
    except Exception as e:
        print(str(e))
    if action == "retrievalDataFromFile":
        message = func_retrieval(action_input)
        if isinstance(message, dict):
            message = message["retrieved_info"]
            # ["retrieved_info"]
    else:
        message = func(action_input, openapi_spec, path, method, retrieval_available, base_url)
    logging.info("observation")
    logging.info(message)
    logging.info("-------------------------------------------------------------------------------------------------")
    # print("observation", message)
    # print("-------------------------------------------------------------------------------------------------")
    information = [[action, action_input, output], message]
    return message, False, information

def load_model_and_tokenizer(model_path):
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="right", 
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = "auto",
    )

    model.requires_grad_(False) 
    model.eval()
    model.generation_config.do_sample = False
    return model, tokenizer

@torch.inference_mode()
def generate_stream_llama(
    model, tokenizer, instruction, max_new_tokens = 256, device="cuda", context_len=8192, force_generate=False
):
    
    fin_input = instruction
    
    stop_token_ids = []
    stop_token_ids.append(tokenizer.eos_token_id)
    
    input_ids = tokenizer(fin_input).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    probs = []
    past_key_values = out = None
    logits=[]
    for i in range(max_new_tokens):
        if i == 0:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values
        
        last_token_logits = logits[0, -1, :]


        token = int(torch.argmax(last_token_logits))
        probs.append(torch.softmax(last_token_logits, dim=-1)[token].item())
        

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False
        if i == 0 and force_generate:
            stopped = False
        if i == max_new_tokens - 1 or stopped:
            
            tmp_output_ids = output_ids[input_echo_len:]
            # clean
            del past_key_values, out
            gc.collect()
            torch.cuda.empty_cache()
            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            logging.info("output")
            logging.info(output)
            # print("model output:", output)
            
            return output

        if stopped:
            break
    tmp_output_ids = output_ids[input_echo_len:]
    output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
    logging.info("output")
    logging.info(output)
    # print("model output", output)

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    return output


default_system = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
                    "Additional, you can utilize external tools to help handle the user's query if needed. "
                    )
                    
def test_single_scene(model, tokenizer, test_list, server_url, enable_getDetails=True):
    # server_url = "http://127.0.0.1:5678"
    # below to get the testing prompt
    openapi_spec = load_openapi_spec(test_list["Documentation"], replace_refs=True)
    components_descriptions = escape(test_list["Function_Description"]["components"])
    tools = [GetDetailsTool()] if not enable_getDetails else []
    for ext_tool in test_list.get("external_tools", []):
        tools.append(tool_projection[ext_tool]())

    for idx, func_name in enumerate(test_list["Function_Projection"]):
        description = escape(test_list["Function_Description"][func_name])
        if idx == len(test_list["Function_Projection"]) - 1:
            description += components_descriptions
        path, method = test_list["Function_Projection"][func_name]
        tools.append(Tool(
            base_url=server_url + "/" + test_list["Name"] if server_url else None,
            func_name=func_name,
            openapi_spec=openapi_spec,
            path=path,
            method=method,
            description=description,
            retrieval_available="retrieval" in test_list.get("external_tools", [])
        ))
    
    prompt = CustomZeroShotAgent.create_prompt(
        tools,
        prefix=test_prompt_v1["prefix"],
        suffix=test_prompt_v1["suffix"],
        format_instructions=test_prompt_v1["format_instructions"],
        input_variables=["input", "agent_scratchpad"]
    )

    

    openapi_spec = json.loads(test_list["Documentation"])
    function_proj = test_list["Function_Projection"]
    answer = []
    for instruction in tqdm(test_list["Instructions"]):
        prefix = prompt.format(input=instruction, agent_scratchpad="")
        if len(test_list.get("Authentication", [])) > 0:
            instruction += "\nAuthentication information: " + \
                    " ".join([f"{k}={v}" for k, v in test_list["Authentication"].items()])

        tool = prefix.split("You have access to the following tools:")[1].split("The chat follows this format:")[0]
        chat_prompt = "The chat follows this format:" + prefix.split("The chat follows this format:")[1].split("\n\nBegin!\n\n")[0]
        prompt_action = chat_prompt[chat_prompt.find("ASSISTANT Action: ")+len("ASSISTANT Action: "):chat_prompt.find("ASSISTANT Action Input:")]
        # chat_prompt = chat_prompt.replace("USER", 'user').replace('ASSISTANT Thought', 'Thought').replace('ASSISTANT Action', 'Action').replace('ASSISTANT Observation', 'Tool calling observation').replace('ASSISTANT Response', 'Response').replace("Observation", "observation")
        chat_prompt = """The chat follows this format:\nbased on the user's question you need to generate as follows:\nThought: the assistant's inner thought about what to do next\nAction: the action to take, must be one of [getDetails, sendHttpRequest, getClientRequestData, testProxyHeaders, simulateStatusCode].\nAction Input: the input for the action, in JSON format.\nAnd the result of the action will be provided\n... (this Thought/Action/Action Input can repeat N times)\nIf the above process is sufficient for you to complete the user's question, you can generate as follows: Thought: summarize the information gathered\nResponse: the final response to the user.\nAnd you can repeat the above process to handle user's next question.\n\nBegin!\n\n"""
        # chat_prompt = chat_prompt.replace("USER", 'user').replace('ASSISTANT Thought', 'Thought').replace('ASSISTANT Action', 'Action').replace('ASSISTANT Observation', 'observation').replace('ASSISTANT Response', 'Response').replace("Observation", "observation")
        # print(chat_prompt)
        chat_prompt = chat_prompt.replace("the action to take, must be one of [getDetails, sendHttpRequest, getClientRequestData, testProxyHeaders, simulateStatusCode].\n", 
                                          prompt_action)
        input = "<s>[INST] system:" + default_system + "\n\n" + "API description:" + tool + chat_prompt + " [/INST]" + \
                "[INST] " + instruction + " [/INST]" + \
                ""
        
        # instruction
        all_output = {}
        all_output["input"] = instruction
        all_output["intermediate_steps"] = []
        logging.info("instruction:") 
        logging.info(input)
        logging.info("--------------------------------------------------------------------------------------------")
        # print("instruction:", input)
        # print("------------------------------------------------------------------------------------------------")
        try_num = 0
        while try_num < 10:
            output = generate_stream_llama(model, tokenizer, input)
            logging.info("----------------------------------------------")
            # print("------------------------------------------------------------------------------------------------")
            message, finish, information = parse_output(output, openapi_spec, function_proj,retrieval_available="retrieval" in test_list.get("external_tools", []),
                                           base_url=server_url + "/" + test_list["Name"] if server_url else None)
            # add the output
            input += output + "</s>"
            # add the observation 
            input += "[INST] observation:" + message + "[/INST]" + \
                    ""
            
                    
            # all_output["intermediate_steps"].append([["action": "","action input": ,"observation":""], message])
            if finish:
                all_output["output"] = information[1]
                all_output["Final_Thought"] = information[0]
                input += output
                break
            else:
                all_output["intermediate_steps"].append(information)
            try_num += 1
        if try_num == 10 and 'output' not in all_output.keys():
            all_output["output"] = ""
            all_output["Final_Thought"] = ""
        try:
            json.dumps(all_output, ensure_ascii=4)
        except json.JSONDecodeError:
            output = str(output)
        except Exception as e:
            logging.error(e)
            output = {"error": str(e)}  
        answer.append(all_output)
    return answer

if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-api", "--api_data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("-out", "--output_dir", type=str, default="")
    parser.add_argument("-llm", type=str, default=None)
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5678")
    parser.add_argument("--output_prefix", type=str, default="api_data")
    parser.add_argument("--agent_prompt", type=str, default="train_v2")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--real", action="store_true", default=False)
    parser.add_argument("--without_getDetails", action="store_true", default=False)
    args = parser.parse_args()
    if not os.path.exists("./log/"+ args.model_path.split("/")[-1] + ".json".replace('.json', '.log')):
        file = open("./log/"+ args.model_path.split("/")[-1] + ".json".replace('.json', '.log'), "w")
        file.close()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename="./log/"+ args.model_path.split("/")[-1] + ".json".replace('.json', '.log'), filemode='w')
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    api_data = json.load(open(args.api_data_path, "r"))

    model_name = args.model_path.split("/")[-1]
    final_output_path = os.path.join(args.output_dir, f"{model_name}.json")

    if not os.path.exists(final_output_path):
        os.system(r"torch {}".format(final_output_path))
    for api_idx, api in tqdm(enumerate(api_data)): # [:1])):
        print(api_idx)
        api["Instances"] = []
        if "Instructions" not in api or len(api["Instructions"]) == 0:
            continue
        openapi_spec = load_openapi_spec(api["Documentation"])
        input_valid, output_valid = analyze_openapi_spec(openapi_spec)
        if input_valid and output_valid:
            Answers = test_single_scene(model, tokenizer, api, args.server_url)

            api["Instances"] = Answers
            json.dump(
                api_data, 
                open(final_output_path, "w", encoding="utf-8"), 
                indent=4, 
                ensure_ascii=False
            )


