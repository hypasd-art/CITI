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

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import PreTrainedModel, PreTrainedTokenizer

def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight
    
def resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""
    Resize token embeddings.
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
            _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

        print("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))


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
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map = "auto",
    )
    if peft_path != None:
        from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
        print("------------------loading from peft-----------------------")

        model = PeftModel.from_pretrained(model, peft_path)
    # resize_embedding_layer(model, tokenizer)

    model.requires_grad_(False) 
    model.eval()
    model.generation_config.do_sample = False
    return model, tokenizer



@torch.inference_mode()
def generate_stream_phi(
    model, tokenizer, instruction, max_new_tokens = 128, device="cuda", context_len=8192, force_generate=False
):
    fin_input = instruction
    
    stop_token_ids = []
    stop_token_ids.append(tokenizer.eos_token_id)
    stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|end|>"))

    
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
        tokenizer.convert_tokens_to_ids("<|end|>")
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
    for test_dir in ['level-1-api', 'level-1-response', 'level-2-api', 'level-2-response', 'level-3-batch-inf-icl', 'level-3-batch-inf-response',]:
        data_dir = "./test-data-phi3-instruct/" + test_dir +".json"
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
                    model_output  = generate_stream_phi(model, tokenizer, content)
                    pred.append({"file": data["file"], "id": data["id"], "instrucion": data["instruction"], "input": data["input"], "output":data["expected_output"], "pred": model_output})
                elif test_lv == "v2":
                    model_output = generate_stream_phi(model, tokenizer, content)
                    pred.append({"file": data["file"], "id": data["id"], "instrucion": data["instruction"], "input": data["input"], "output":data["expected_output"], "pred": model_output})
                else:
                    model_output = generate_stream_phi(model, tokenizer, content)
                    pred.append({"sample_id": data["sample_id"], "api_id": data["api_id"], "instrucion": data["instruction"], "input": data["input"], "output":data["output"], "pred": model_output})

            elif dialog_test_enabled:
                content = data["instruction"] + data["input"]
                if test_lv == "v1" or "v2" == test_lv:
                    model_output = generate_stream_phi(model, tokenizer, content)
                    pred.append({"file": data["file"], "id": data["id"], "instrucion": data["instruction"], "input": data["input"], "output":data["expected_output"], "pred": model_output})
                else:
                    model_output = generate_stream_phi(model, tokenizer, content)
                    pred.append({"sample_id": data["sample_id"], "api_id": data["api_id"], "instrucion": data["instruction"], "input": data["input"], "output":data["output"], "pred": model_output})

        p = json.dumps(pred)
        with open(evaluation_path, 'w') as f:
            f.write(p)
            
        print(f"saved in {evaluation_path}")

import fire
if __name__ == '__main__':
    fire.Fire(main) 

