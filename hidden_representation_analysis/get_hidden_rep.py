import torch
import os
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import json
import numpy
import gc
import time
import pickle
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
import random

random.seed(666)



@torch.inference_mode()
def generate_stream(
    model, tokenizer, instruction, max_new_tokens = 1, device="cuda", context_len=2048
):

    input_ids = tokenizer(instruction).input_ids
    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]


    out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
    # clean
    del out
    gc.collect()
    torch.cuda.empty_cache()
    
    return 
    


def main(
    base_model: str = "", 
    tokenizer_path: str = "",
    output_path: str = "./output/"
):
    if tokenizer_path == "":
        tokenizer_path = base_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, padding_side="right",)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

    model.eval()
    print(model)

    features = {}
    mean_features = {}
    names =set()
    if not os.path.exists(output_path+base_model.split("/")[-1]+"/"):
        os.mkdir(output_path+base_model.split("/")[-1]+"/")
    

    with open("./all_sample_data.pickle", "rb") as file:
        all_data = pickle.load(file)
    for ids, (name, data) in enumerate(all_data.items()):
        feature = {}
        mean_feature = {}
        fea_hooks = get_feas_by_hook(model)
        print("-"*100)
        print(name)

        for instruction in tqdm(data):
            generate_stream(model, tokenizer, instruction)

        for fea_hook in fea_hooks:
            names.add(fea_hook.name)
            
            feature[fea_hook.name] = []

            for fea in fea_hook.fea:
                # print(fea.shape)
                feature[fea_hook.name].append(fea)

            fea_to_add = torch.stack(feature[fea_hook.name]) # .numpy().tolist()
            print(fea_to_add.shape)
            feature[fea_hook.name] = fea_to_add.numpy().tolist()
            mean_feature[fea_hook.name] = fea_to_add.mean(dim=0)
        features[name] = feature
        mean_features[name] = mean_feature
    
    with open(output_path+base_model.split("/")[-1]+'/fea.json', 'w') as f:  
        json.dump(features, f)

    

    
    

class HookTool: 
    def __init__(self, n):
        self.name = n
        self.fea = [] 
        self.new_sent = True

    def hook_fun(self, module, fea_in, fea_out):
        self.fea.append(fea_in[0][0,-1,:].cpu())

def get_feas_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if "mlp.up_proj" in n or "self_attn.q_proj" in n:
            cur_hook = HookTool(n)
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks


if __name__ == "__main__":
    # main(base_model="/mnt/publiccache/yphao/model/apibank/apibank_llama3_instruct_full", tokenizer_path="/mnt/publiccache/yphao/model/apibank/apibank_llama3_instruct_full")
    # main(base_model="/mnt/publiccache/yphao/model/apibank/code_alpaca_llama3_instruct_full", tokenizer_path="/mnt/publiccache/yphao/model/apibank/code_alpaca_llama3_instruct_full")
    main(base_model="/mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct", tokenizer_path="/mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct")
    