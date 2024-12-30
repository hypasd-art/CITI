# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

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

__version__ = "0.3.0.dev0"

from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING, get_peft_config, get_peft_model
from .peft_model import (
    PeftModel_local,
    PeftModelForCausalLM_local,
    PeftModelForSeq2SeqLM_local,
    PeftModelForSequenceClassification_local,
    PeftModelForTokenClassification_local,
)
from .tuners import (
    LoraConfig_local,
    LoraModel_local,
    PrefixEncoder_local,
    PrefixTuningConfig_local,
    PromptEmbedding_local,
    PromptEncoder_local,
    PromptEncoderConfig_local,
    PromptEncoderReparameterizationType_local,
    PromptTuningConfig_local,
    PromptTuningInit_local,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    PeftConfig_local,
    PeftType_local,
    PromptLearningConfig_local,
    TaskType_local,
    bloom_model_postprocess_past_key_value,
    get_peft_model_state_dict_local,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict_local,
    shift_tokens_right,
)