# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from .adapter import init_adapter
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments
########################################################################################
from .modeling_llama_moe.llama.modeling_llama import LlamaForCausalLM_MOLoRA, LlamaConfig
from .modeling_llama_moe.mistral.modeling_mistral import MistralForCausalLM_MOLoRA, MistralConfig
from .modeling_llama_moe.phi.modeling_phi3 import Phi3ForCausalLM_MOLoRA, Phi3Config
from .modeling_llama_moe.peft_local.peft_model import PeftModel_local
from .modeling_llama_moe.peft_local import LoraConfig_local, TaskType_local, get_peft_model, PeftModel_local, get_peft_model_state_dict_local

from transformers.integrations import is_deepspeed_zero3_enabled

import json
def load_dict_from_file(filename='data.json'):
    """
    reading the molora allocation strategy
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            dictionary = json.load(file)
        print(f"loading molora allocation from {filename}")
        return dictionary
    except IOError as e:
        print(f"can't read {e}")

def load_dict_from_file_unfreeze_module(filename='data.json'):
    """
    reading the freeze linear module
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            dictionary = json.load(file)
        print(f"loading freezing module from {filename}")
        return dictionary
    except IOError as e:
        print(f"can't read {e}")
#######################################################################################

logger = get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)

    if model_args.visual_inputs:
        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
            setattr(processor, "tokenizer", tokenizer)
        except Exception:
            raise ValueError(
                "This multimodal LLM is not supported.\n"
                "Download LLaVA-1.5 models from: https://huggingface.co/llava-hf\n"
                "Download Yi-VL models from: https://huggingface.co/BUAADreamer"
            )
    else:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    ###############################################################################
    if model_args.model_type == "CITI":
        if "Llama" in model_args.model_name_or_path:
            config = LlamaConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        elif "Phi" in model_args.model_name_or_path:
            config = Phi3Config.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        elif "Mistral" in model_args.model_name_or_path:
            config = MistralConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        else:
            raise NameError
        # config = LlamaConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        
        molora_dict = load_dict_from_file(model_args.moelora_allocation)
        config.molora_dict = molora_dict
        print("---------------------------the components need to add molora---------------------------")
        print(config.molora_dict)
        return config
    else:
    ###############################################################################
        return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        elif model_args.visual_inputs:
            model = AutoModelForVision2Seq.from_pretrained(**init_kwargs)
        elif model_args.train_from_scratch:
            model = AutoModelForCausalLM.from_config(config)
        else:
            ################################################################################################
            if model_args.model_type == "CITI":
                if "Llama" in model_args.model_name_or_path:
                    model = LlamaForCausalLM_MOLoRA.from_pretrained(
                        **init_kwargs,
                    )
                elif "Phi" in model_args.model_name_or_path:
                    model = Phi3ForCausalLM_MOLoRA.from_pretrained(
                        **init_kwargs,
                    )
                elif "Mistral" in model_args.model_name_or_path:
                    model = MistralForCausalLM_MOLoRA.from_pretrained(
                        **init_kwargs,
                    )
                else:
                    raise NameError
                model.enable_input_require_grads()
                model.config.use_cache = False
                
                # training the lora moe adapter
                if model_args.training_stage == "stage_1": 
                    print("          Init new peft model             ") 
                    moe_target_modules = load_dict_from_file(model_args.moelora_allocation) 
                    if model_args.lora_allocation == "":
                        lora_target_modules = []
                    else:
                        lora_target_modules = load_dict_from_file(model_args.lora_allocation)

                    peft_config = LoraConfig_local(
                        task_type=TaskType_local.CAUSAL_LM,
                        inference_mode=False,
                        moe_target_modules = moe_target_modules,
                        lora_target_modules = lora_target_modules,
                        molora_dict = None,
                        r=model_args.moelora_rank, 
                        num_select=model_args.moelora_select,
                        lora_alpha=model_args.moelora_alpha,
                        lora_dropout=model_args.moelora_dropout,
                        lora_nums=model_args.lora_nums,
                        blc_weight=model_args.blc_weight,
                        blc_alpha=model_args.blc_alpha,
                        moe_type=model_args.moe_type,
                        )
                    
                    model = get_peft_model(model, peft_config)
                    for name, param in model.named_parameters():  
                        param.requires_grad = False

                    for name, param in model.named_parameters():  
                        if ".lora_route_network." in name:  
                            param.requires_grad = True 
                elif model_args.training_stage == "stage_2":
                    print("          load trained peft model             ")
                    model = PeftModel_local.from_pretrained(model, model_args.peft_path)
                    for name, param in model.named_parameters():  
                        param.requires_grad = False  
                        
                    for name, param in model.named_parameters():  
                        if "lora_" in name:  
                            param.requires_grad = True 
                elif model_args.training_stage == "stage_3":
                    print("          load trained peft model             ")
                    enable_blc_weight = True if model_args.blc_weight > 1e-5 else False
                    model = PeftModel_local.from_pretrained(model, model_args.peft_path, enable_blc_weight)
                    for name, param in model.named_parameters():  
                        param.requires_grad = False 

                        
                    if model_args.unfreeze_module != None:
                        unfreeze_module = load_dict_from_file_unfreeze_module(model_args.unfreeze_module)
                        print("-------------------unfreeze module in training stage 3-------------------")
                        for module_name in unfreeze_module:
                            for name, param in model.named_parameters():  
                                if module_name in name:  
                                    print(name)
                                    param.requires_grad = True  
                elif model_args.training_stage == "wo_router_1":
                    print("          Init new peft model             ") 
                    moe_target_modules = load_dict_from_file(model_args.moelora_allocation) 
                    if model_args.lora_allocation == "":
                        lora_target_modules = []
                    else:
                        lora_target_modules = load_dict_from_file(model_args.lora_allocation)

                    peft_config = LoraConfig_local(
                        task_type=TaskType_local.CAUSAL_LM,
                        inference_mode=False,
                        moe_target_modules = moe_target_modules,
                        lora_target_modules = lora_target_modules,
                        molora_dict = None,
                        r=model_args.moelora_rank, 
                        num_select=model_args.moelora_select,
                        lora_alpha=model_args.moelora_alpha,
                        lora_dropout=model_args.moelora_dropout,
                        lora_nums=model_args.lora_nums,
                        blc_weight=model_args.blc_weight,
                        blc_alpha=model_args.blc_alpha,
                        moe_type=model_args.moe_type,
                        )
                    
                    model = get_peft_model(model, peft_config)
                    for name, param in model.named_parameters():  
                        param.requires_grad = False

                    for name, param in model.named_parameters():  
                        if "lora_" in name:  
                            param.requires_grad = True 
                elif model_args.training_stage == "wo_router_2":
                    print("          load trained peft model             ")
                    enable_blc_weight = True if model_args.blc_weight > 1e-5 else False
                    model = PeftModel_local.from_pretrained(model, model_args.peft_path, enable_blc_weight)
                    for name, param in model.named_parameters():  
                        param.requires_grad = False 

                        
                    if model_args.unfreeze_module != None:
                        unfreeze_module = load_dict_from_file_unfreeze_module(model_args.unfreeze_module)
                        print("-------------------unfreeze module in training stage 3-------------------")
                        for module_name in unfreeze_module:
                            for name, param in model.named_parameters():  
                                if module_name in name:  
                                    print(name)
                                    param.requires_grad = True  
                # elif model_args.training_stage == "wo_Lr":
                    
                else:
                    raise
            else:
            ################################################################################################
                model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

                if model_args.unfreeze_module != None:
                    for name, param in model.named_parameters():  
                        param.requires_grad = False 
                    unfreeze_module = load_dict_from_file_unfreeze_module(model_args.unfreeze_module)
                    print("-------------------unfreeze module in training process-------------------")
                    for module_name in unfreeze_module:
                        for name, param in model.named_parameters():  
                            if module_name in name:  
                                print(name)
                                param.requires_grad = True  

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        if model_args.model_type != "CITI":
            register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )
    print(model)
    return model
