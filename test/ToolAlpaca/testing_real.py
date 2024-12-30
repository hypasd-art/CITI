import logging
import requests
from datetime import datetime
import torch
import gc
from utils import load_openapi_spec, escape
from agent.tools import Tool, GetDetailsTool
from agent.agent_prompts import test_prompt_v1
from agent.custom_agent import CustomZeroShotAgent

logger = logging.getLogger(__name__)


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

    logger.debug("request url: {url}")

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
    
    logger.debug("url: {response.request.url}")
    logger.debug("body: {response.request.body}")
    logger.debug("headers: {response.request.headers}")

    return response

def func(params, openapi_spec, path, method, base_url=None):
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
    retry_times = 0
    while retry_times < 3:
        try:
            response = call_api_function(
                input_params=params,
                openapi_spec=openapi_spec,
                path=path,
                method=method,
                base_url=base_url
            )
        except ValueError as e:
            return str(e)
        message = f"Status Code: {response.status_code}. Response: {response.text}"
        if response.status_code >= 500:
            retry_times += 1
            continue
        break
    if not 200 <= response.status_code < 300:
        # message += ". You can try to change the input or call another function. "
        message += ". You should choose one of: (1) change the input and retry; (2) return the 'Final Answer' and explain what happened; (You must choose this one when the error occurs more than 3 times.) (3) call another function."
        
    max_output_len = 1000
    if len(message) > max_output_len:
        
        message = message[:max_output_len]
    return message

def parse_output(output, openapi_spec, function_proj):
    if "Response:" in output:
        try:
            assert "Action:" not in output[output.find("Response:"):]
            assert "Action Input:" not in output[output.find("Response:"):]
        except:
            message = "Invalid output. Please do not call tools after give the final response" #'Action Input' "
            return message, False
        return "", True
    thought = output.split("Action:")[0]
    action = output.split("Action:")[1].split("Action Input:")[0]
    action_input = output.split("Action:")[1].split("Action Input:")[1]
    path = function_proj[action][0]
    method = function_proj[action][1]
    message = func(action_input, openapi_spec, path, method)
    return message, False

def load_model_and_tokenizer(model_path):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="right", 
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = "auto",
    )
    model = model.float()

    model.requires_grad_(False) 
    model.eval()
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
            print(output)
            
            return output

        if stopped:
            break
    tmp_output_ids = output_ids[input_echo_len:]
    output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
    print(output)

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    return output



default_system=("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
                "Additional, you can utilize external tools to help handle the user's query if needed. "
                "And you can generate a response based on the user's utterance and API Requests. "
                )

def test_single_scene(model, tokenizer, test_list):
    server_url = "http://127.0.0.1:5678"
    openapi_spec = load_openapi_spec(test_list["Documentation"])
    components_descriptions = escape(test_list["Function_Description"]["components"])
    tools = [GetDetailsTool()]
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
    
    # tools.append(NoTool())

    
    prompt = CustomZeroShotAgent.create_prompt(
        tools,
        prefix=test_prompt_v1["prefix"],
        suffix=test_prompt_v1["suffix"],
        format_instructions=test_prompt_v1["format_instructions"],
        input_variables=["input", "agent_scratchpad"]
    )

    

    openapi_spec = test_list["Documentation"]
    function_proj = test_list["Function_Projection"]
    for instruction in test_list["Instructions"]:
        prefix = prompt.format(input=instruction, agent_scratchpad="")

        tool = prefix.split("You have access to the following tools:")[1].split("The chat follows this format:")[0]
        chat_prompt = "The chat follows this format:" + prefix.split("The chat follows this format:")[1].split("\n\nBegin!\n\n")[0]

        input = "<|start_header_id|>system<|end_header_id|>\n\n" + default_system + "\n\n" + "API description:" + tool + chat_prompt + "<|eot_id|>" + \
                "<|start_header_id|>user<|end_header_id|>\n\n" + instruction + "<|eot_id|>" + \
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # instruction
        while True:
            output = generate_stream_llama(model, tokenizer, input)
            message, finish = parse_output(output, openapi_spec, function_proj)
            input += output
            input += "<|start_header_id|>tool<|end_header_id|>\n\n" + message + "<|eot_id|>" + \
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if finish:
                input += output
                break
        try:
            json.dumps(all_output, ensure_ascii=4)
        except json.JSONDecodeError:
            output = str(output)
        except Exception as e:
            logger.error(e)
            output = {"error": str(e)}  


import json
if __name__ == "__main__":

    document = "{\"openapi\": \"3.0.1\", \"info\": {\"title\": \"Nager.Date API - V3\", \"description\": \"Nager.Date is open source software. If you would like to support the project you can award a GitHub star \\u2b50 or much better <a href='https://github.com/sponsors/nager'>start a sponsorship</a>\", \"contact\": {\"name\": \"Nager.Date on GitHub\", \"url\": \"https://github.com/nager/Nager.Date\"}, \"license\": {\"name\": \"MIT License\", \"url\": \"https://github.com/nager/Nager.Date/blob/master/LICENSE.md\"}, \"version\": \"v3\"}, \"servers\": [{\"url\": \"https://date.nager.at/\"}], \"paths\": {\"/api/v3/CountryInfo/{countryCode}\": {\"get\": {\"tags\": [\"Country\"], \"summary\": \"Get country info for the given country\", \"operationId\": \"CountryCountryInfo\", \"parameters\": [{\"name\": \"countryCode\", \"description\": \"Two-character represented country code. For instance, CN or cn represents China.\", \"in\": \"path\", \"required\": true, \"style\": \"simple\", \"explode\": false, \"schema\": {\"type\": \"string\"}}], \"responses\": {\"200\": {\"description\": \"Success\", \"content\": {\"text/plain\": {\"schema\": {\"$ref\": \"#/components/schemas/CountryInfoDto\"}}, \"application/json\": {\"schema\": {\"$ref\": \"#/components/schemas/CountryInfoDto\"}}, \"text/json\": {\"schema\": {\"$ref\": \"#/components/schemas/CountryInfoDto\"}}}}}}}, \"/api/v3/AvailableCountries\": {\"get\": {\"tags\": [\"Country\"], \"summary\": \"Get all available countries\", \"operationId\": \"CountryAvailableCountries\", \"responses\": {\"200\": {\"description\": \"Success\", \"content\": {\"text/plain\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/CountryV3Dto\"}}}, \"application/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/CountryV3Dto\"}}}, \"text/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/CountryV3Dto\"}}}}}}}}, \"/api/v3/LongWeekend/{year}/{countryCode}\": {\"get\": {\"tags\": [\"LongWeekend\"], \"summary\": \"Get long weekends for a given country\", \"operationId\": \"LongWeekendLongWeekend\", \"parameters\": [{\"name\": \"year\", \"in\": \"path\", \"required\": true, \"style\": \"simple\", \"explode\": false, \"schema\": {\"type\": \"integer\", \"format\": \"int32\"}}, {\"name\": \"countryCode\", \"in\": \"path\", \"description\": \"Two-character represented country code. For instance, CN or cn represents China.\", \"required\": true, \"style\": \"simple\", \"explode\": false, \"schema\": {\"type\": \"string\"}}], \"responses\": {\"200\": {\"description\": \"Success\", \"content\": {\"text/plain\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/LongWeekendV3Dto\"}}}, \"application/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/LongWeekendV3Dto\"}}}, \"text/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/LongWeekendV3Dto\"}}}}}}}}, \"/api/v3/PublicHolidays/{year}/{countryCode}\": {\"get\": {\"tags\": [\"PublicHoliday\"], \"summary\": \"Get public holidays\", \"operationId\": \"PublicHolidayPublicHolidaysV3\", \"parameters\": [{\"name\": \"year\", \"in\": \"path\", \"required\": true, \"style\": \"simple\", \"explode\": false, \"schema\": {\"type\": \"integer\", \"format\": \"int32\"}}, {\"name\": \"countryCode\", \"in\": \"path\", \"description\": \"Two-character represented country code. For instance, CN or cn represents China.\", \"required\": true, \"style\": \"simple\", \"explode\": false, \"schema\": {\"type\": \"string\"}}], \"responses\": {\"200\": {\"description\": \"Public holidays\", \"content\": {\"text/plain\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}, \"application/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}, \"text/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}}}, \"400\": {\"description\": \"Validation failure\"}, \"404\": {\"description\": \"CountryCode is unknown\"}}}}, \"/api/v3/IsTodayPublicHoliday/{countryCode}\": {\"get\": {\"tags\": [\"PublicHoliday\"], \"summary\": \"Is today a public holiday\", \"description\": \"The calculation is made on the basis of UTC time to adjust the time please use the offset.<br />\\r\\nThis is a special endpoint for `curl`<br /><br />\\r\\n200 = Today is a public holiday<br />\\r\\n204 = Today is not a public holiday<br /><br />\\r\\n`STATUSCODE=$(curl --silent --output /dev/stderr --write-out \\\"%{http_code}\\\" https://date.nager.at/Api/v2/IsTodayPublicHoliday/AT)`<br /><br />\\r\\n`if [ $STATUSCODE -ne 200 ]; then # error handling; fi`\", \"operationId\": \"PublicHolidayIsTodayPublicHoliday\", \"parameters\": [{\"name\": \"countryCode\", \"in\": \"path\", \"description\": \"Two-character represented country code. For instance, CN or cn represents China.\", \"required\": true, \"style\": \"simple\", \"explode\": false, \"schema\": {\"type\": \"string\"}}, {\"name\": \"countyCode\", \"in\": \"query\", \"required\": false, \"style\": \"form\", \"explode\": true, \"schema\": {\"type\": \"string\"}}, {\"name\": \"offset\", \"in\": \"query\", \"description\": \"utc timezone offset\", \"required\": false, \"style\": \"form\", \"explode\": true, \"schema\": {\"maximum\": 12, \"minimum\": -12, \"type\": \"integer\", \"format\": \"int32\", \"default\": 0}}], \"responses\": {\"200\": {\"description\": \"Today is a public holiday\"}, \"204\": {\"description\": \"Today is not a public holiday\"}, \"400\": {\"description\": \"Validation failure\"}, \"404\": {\"description\": \"CountryCode is unknown\"}}}}, \"/api/v3/NextPublicHolidays/{countryCode}\": {\"get\": {\"tags\": [\"PublicHoliday\"], \"summary\": \"Returns the upcoming public holidays for the next 365 days for the given country\", \"operationId\": \"PublicHolidayNextPublicHolidays\", \"parameters\": [{\"name\": \"countryCode\", \"in\": \"path\", \"description\": \"Two-character represented country code. For instance, CN or cn represents China.\", \"required\": true, \"style\": \"simple\", \"explode\": false, \"schema\": {\"type\": \"string\"}}], \"responses\": {\"200\": {\"description\": \"Success\", \"content\": {\"text/plain\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}, \"application/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}, \"text/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}}}}}}, \"/api/v3/NextPublicHolidaysWorldwide\": {\"get\": {\"tags\": [\"PublicHoliday\"], \"summary\": \"Returns the upcoming public holidays for the next 7 days\", \"operationId\": \"PublicHolidayNextPublicHolidaysWorldwide\", \"responses\": {\"200\": {\"description\": \"Success\", \"content\": {\"text/plain\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}, \"application/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}, \"text/json\": {\"schema\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayV3Dto\"}}}}}}}}, \"/api/v3/Version\": {\"get\": {\"tags\": [\"Version\"], \"summary\": \"Get version of the used Nager.Date library\", \"operationId\": \"VersionGetVersion\", \"responses\": {\"200\": {\"description\": \"Success\", \"content\": {\"text/plain\": {\"schema\": {\"$ref\": \"#/components/schemas/VersionInfoDto\"}}, \"application/json\": {\"schema\": {\"$ref\": \"#/components/schemas/VersionInfoDto\"}}, \"text/json\": {\"schema\": {\"$ref\": \"#/components/schemas/VersionInfoDto\"}}}}}}}}, \"components\": {\"schemas\": {\"CountryInfoDto\": {\"type\": \"object\", \"properties\": {\"commonName\": {\"type\": \"string\", \"description\": \"CommonName\", \"nullable\": true}, \"officialName\": {\"type\": \"string\", \"description\": \"OfficialName\", \"nullable\": true}, \"countryCode\": {\"type\": \"string\", \"description\": \"Two-character represented country code. For instance, CN or cn represents China.\", \"nullable\": true}, \"region\": {\"type\": \"string\", \"description\": \"Region\", \"nullable\": true}, \"borders\": {\"type\": \"array\", \"description\": \"Country Borders\", \"nullable\": true, \"items\": {\"$ref\": \"#/components/schemas/CountryInfoDto\"}}}, \"additionalProperties\": false, \"description\": \"CountryInfo Dto\"}, \"CountryV3Dto\": {\"type\": \"object\", \"properties\": {\"countryCode\": {\"type\": \"string\", \"nullable\": true}, \"name\": {\"type\": \"string\", \"nullable\": true}}, \"additionalProperties\": false, \"description\": \"Country\"}, \"LongWeekendV3Dto\": {\"type\": \"object\", \"properties\": {\"startDate\": {\"type\": \"string\", \"description\": \"StartDate\", \"format\": \"date-time\"}, \"endDate\": {\"type\": \"string\", \"description\": \"EndDate\", \"format\": \"date-time\"}, \"dayCount\": {\"type\": \"integer\", \"description\": \"DayCount\", \"format\": \"int32\"}, \"needBridgeDay\": {\"type\": \"boolean\", \"description\": \"NeedBridgeDay\"}}, \"additionalProperties\": false, \"description\": \"Long Weekend\"}, \"PublicHolidayType\": {\"type\": \"string\", \"enum\": [\"Public\", \"Bank\", \"School\", \"Authorities\", \"Optional\", \"Observance\"]}, \"PublicHolidayV3Dto\": {\"type\": \"object\", \"properties\": {\"date\": {\"type\": \"string\", \"description\": \"The date\", \"format\": \"date\"}, \"localName\": {\"type\": \"string\", \"description\": \"Local name\", \"nullable\": true}, \"name\": {\"type\": \"string\", \"description\": \"English name\", \"nullable\": true}, \"countryCode\": {\"type\": \"string\", \"description\": \"ISO 3166-1 alpha-2\", \"nullable\": true}, \"fixed\": {\"type\": \"boolean\", \"description\": \"Is this public holiday every year on the same date\"}, \"global\": {\"type\": \"boolean\", \"description\": \"Is this public holiday in every county (federal state)\"}, \"counties\": {\"type\": \"array\", \"description\": \"ISO-3166-2 - Federal states\", \"nullable\": true, \"items\": {\"type\": \"string\"}}, \"launchYear\": {\"type\": \"integer\", \"description\": \"The launch year of the public holiday\", \"format\": \"int32\", \"nullable\": true}, \"types\": {\"type\": \"array\", \"description\": \"A list of types the public holiday it is valid\", \"nullable\": true, \"items\": {\"$ref\": \"#/components/schemas/PublicHolidayType\"}}}, \"additionalProperties\": false, \"description\": \"Public Holiday\"}, \"VersionInfoDto\": {\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\", \"nullable\": true}, \"version\": {\"type\": \"string\", \"nullable\": true}}, \"additionalProperties\": false}}}}"
    openapi_spec = json.loads(document)
    # print(openapi_spec)
    print("paths")
    print(openapi_spec["paths"].keys())
    print("\n\n\n\n\n")
    # print(openapi_spec["paths"][path][method])

    input_params = {"countryCode": "CN"}
    path = "/api/v3/CountryInfo/{countryCode}"
    method = 'get'
    print(openapi_spec["paths"][path][method])
    print("\n\n\n\n\n")
    response = call_api_function(input_params, openapi_spec, path, method)
    print("**************************response*****************************")
    print(response)
    message = f"Status Code: {response.status_code}. Response: {response.text}"
    if response.status_code >= 500:
        print("status_code>500")
    if not 200 <= response.status_code < 300:
        # message += ". You can try to change the input or call another function. "
        message += ". You should choose one of: (1) change the input and retry; (2) return the 'Final Answer' and explain what happened; (You must choose this one when the error occurs more than 3 times.) (3) call another function."
    max_output_len = 2000
    if len(message) > max_output_len:
        message = message[:max_output_len]
    print(message)