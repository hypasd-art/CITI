# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig_local, PeftType_local, transpose


@dataclass
class LoraConfig_local(PeftConfig_local):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    moe_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    blc_alpha: int = field(default=None, metadata={"help": "Alpha of blcloss"})
    blc_weight: float = field(default=None, metadata={"help": "Weight of blcloss"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    molora_dict: dict = field(
        default=None,
        metadata={"help": "the molora dynamic allocation strategy"}
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    ########################################################
    num_select: Optional[int] = field(
        default=1,
        metadata={"help": "the expert to choose is using topk strategy"}
    )
    
    moe_type: Optional[str] = field(
        default="backbone",
        metadata={"help": "the molora type to choose"}
    )
    ########################################################



    def __post_init__(self):
        self.peft_type = PeftType_local.LORA


class LoraModel_local(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model): # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "blc_alpha": self.peft_config.blc_alpha,
            "blc_weight": self.peft_config.blc_weight,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        base_r = self.peft_config.r
        key_list = [key for key, _ in self.model.named_modules()]
        try:
            assert len(set(self.peft_config.moe_target_modules).intersection(self.peft_config.lora_target_modules)) == 0
        except:
            print(set(self.peft_config.moe_target_modules).intersection(self.peft_config.lora_target_modules))
            raise
        for key in key_list:
            if isinstance(self.peft_config.moe_target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.moe_target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.moe_target_modules)
            if target_module_found: 
                # -------------------> here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                ##################################################################################################
                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    
                    
                    if self.peft_config.moe_type == "backbone":
                        new_module = moe_Linear_backbone(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        raise NameError
                ##################################################################################################


                self._replace_module(parent, target_name, new_module, target)
        if self.peft_config.lora_target_modules != None:
            for key in key_list:
                if isinstance(self.peft_config.lora_target_modules, str):
                    target_module_found = re.fullmatch(self.peft_config.lora_target_modules, key)
                else:
                    target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.lora_target_modules)
                if target_module_found: 
                    # -------------------> here
                    if not is_target_modules_in_base_model:
                        is_target_modules_in_base_model = True
                    parent, target, target_name = self._get_submodules(key)
                    bias = target.bias is not None

                    ##################################################################################################
                    if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                        new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                    ##################################################################################################


                    self._replace_module(parent, target_name, new_module, target)
                    
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError



class LoraLayer:
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, in_features, out_features, **kwargs) -> None:
        self.r = 0
        self.lora_alpha = 0
        self.scaling = 0
        self.lora_dropout = None
        self.lora_A = None
        self.lora_B = None
        
        self.kwargs = kwargs

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, r, lora_alpha, lora_dropout, init_lora_weights, 
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r = r
        self.lora_alpha = lora_alpha
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # Actual trainable parameters
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        
        self.scaling = lora_alpha / r
        self.reset_lora_parameters(init_lora_weights)

    def reset_lora_parameters(self, init_lora_weights):
        if init_lora_weights is False:
            return

        if init_lora_weights is True:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.r)
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")
        nn.init.zeros_(self.lora_B.weight)

class lora_Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features,
        out_features,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features, **kwargs)

        self.update_layer(
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result
        

    
class moe_Linear_backbone(nn.Linear):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        
        self.fan_in_fan_out = fan_in_fan_out

        # lora_num + 1 contains the neuron about backbone network
        self.lora_route_network = nn.Linear(in_features, self.lora_num + 1, bias=False)
        self.lora_proj = nn.ModuleList(
                [lora_Linear(in_features, out_features, lora_dropout=lora_dropout, r = r, lora_alpha=lora_alpha) for i in range(self.lora_num)]
            )
        
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        
        # self.reset_parameters()
        nn.init.kaiming_uniform_(self.lora_route_network.weight, a=math.sqrt(5))
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T


    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route_network.train(mode)
        self.lora_proj.train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route_network.eval()
        self.lora_proj.eval()
        
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x: torch.Tensor, task_types=None):
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        route_weight = nn.functional.softmax(self.lora_route_network(x), dim=-1)# bs seq_len dim

        for i in range(self.lora_num):
            # i=0 represent the weight about the backbone
            result = result + torch.unsqueeze(route_weight[:,:,i+1], -1) * self.lora_proj[i](x) 

        blcls = torch.zeros(1)[0] # .to(result.device)
        if self.training:
            if task_types != None:
                # 1: tool; 0: general
                if self.blc_weight != 0:
                    task_types = task_types.view(-1, 1)
                    
                    blcls = self.cv_squared((
                        route_weight.sum(dim=(1)) * torch.where(
                            torch.concat(
                                ((task_types==0), (task_types==1).repeat(1, self.lora_num)), dim=-1
                                ), 1.0-self.blc_alpha, 1.0+self.blc_alpha
                            )
                        ).flatten()
                    ) * self.blc_weight
            else:
                # only balance the weight of the lora if the task types is none
                print("no task types")
                if self.blc_weight != 0:
                    blcls = self.cv_squared(
                        route_weight[:,:,1:].sum(dim=(1)).flatten()
                    ) * self.blc_weight

        return result, blcls, route_weight.sum(0).sum(0)
    
