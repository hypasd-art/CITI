import json
from rouge import Rouge
import os
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import gc


def calculate_rouge_l_score(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score


def load_model_and_tokenizer(model_path, peft_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="right", 
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = "auto",
    )
    if peft_path != None:
        from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
        print("\n\n\nloading from peft\n\n\n")

        model = PeftModel.from_pretrained(model, peft_path)

    model.requires_grad_(False) 
    model.eval()
    model.generation_config.do_sample = False
    return model, tokenizer



@torch.inference_mode()
def generate_stream_llama(
    model, tokenizer, instruction, max_new_tokens = 128, device="cuda", context_len=8192, force_generate=False
):
    fin_input = instruction
    print(fin_input)
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


def generate_local(
    model, tokenizer, instruction, max_new_tokens = 128, device="cuda", context_len=8192, force_generate=False
):
    terminators = [
        tokenizer.eos_token_id,
    ]
    model_inputs = tokenizer([instruction], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response



def main(
        model_path = "/data/yphao/llama/Lynx-7b",
        peft_path = None,
        ):
    model, tokenizer = load_model_and_tokenizer(model_path, peft_path)
    model_path = peft_path if peft_path != None else model_path
    for test_dir in ['level-1-api', 'level-1-response', 'level-2-api', 'level-2-response', 'level-3-batch-inf-icl', 'level-3-batch-inf-response',]:#'level-3-batch-inf-icl', ]:
        data_dir = "./test-data-mistral-instruct/" + test_dir +".json"
        print("\n")
        print(f"test_dir:{data_dir}")
        test_lv = "v1" if "1" in data_dir.split("/")[-1] else ("v2" if "2" in data_dir.split("/")[-1] else "v3")
        print(f"test_lv: {test_lv}")
        print(model_path)
        
        evaluation_path = "predict_result/"+ model_path.split("/")[-1] + "/predict/"+data_dir.split("/")[-1]
        if not os.path.exists("predict_result/"+ model_path.split("/")[-1] + "/predict/"):
            os.makedirs("predict_result/"+ model_path.split("/")[-1] + "/predict/")

        if not os.path.exists(evaluation_path):
            os.system(r"torch {}".format(evaluation_path))

        api_test_enabled = True if "api" in data_dir.split("/")[-1] or "icl" in data_dir.split("/")[-1] else False
        dialog_test_enabled = not api_test_enabled

        pred = []
        

        f = open(data_dir, 'r')
        content = f.read()
        jsonl_datas = json.loads(content)
        for data in tqdm(jsonl_datas, desc='Processing files', ncols=100):
            if api_test_enabled:
                content = data["instruction"] + data["input"]
                if test_lv == "v1":
                    model_output  = generate_stream_llama(model, tokenizer, content)
                    pred.append({"file": data["file"], "id": data["id"], "instrucion": data["instruction"], "input": data["input"], "output":data["expected_output"], "pred": model_output})
                elif test_lv == "v2":
                    model_output = generate_stream_llama(model, tokenizer, content)
                    pred.append({"file": data["file"], "id": data["id"], "instrucion": data["instruction"], "input": data["input"], "output":data["expected_output"], "pred": model_output})
                else:
                    model_output = generate_stream_llama(model, tokenizer, content)
                    pred.append({"sample_id": data["sample_id"], "api_id": data["api_id"], "instrucion": data["instruction"], "input": data["input"], "output":data["output"], "pred": model_output})

            elif dialog_test_enabled:
                content = data["instruction"] + data["input"]
                
                if test_lv == "v1" or "v2" == test_lv:
                    model_output = generate_stream_llama(model, tokenizer, content)
                    pred.append({"file": data["file"], "id": data["id"], "instrucion": data["instruction"], "input": data["input"], "output":data["expected_output"], "pred": model_output})
                else:
                    model_output = generate_stream_llama(model, tokenizer, content)
                    pred.append({"sample_id": data["sample_id"], "api_id": data["api_id"], "instrucion": data["instruction"], "input": data["input"], "output":data["output"], "pred": model_output})

        p = json.dumps(pred)
        with open(evaluation_path, 'w') as f:
            f.write(p)
            
        print(f"saved in {evaluation_path}")

import fire
if __name__ == '__main__':
    fire.Fire(main) 

