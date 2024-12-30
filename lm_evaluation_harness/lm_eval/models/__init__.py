from . import gpt2
from . import gpt3
from . import opt
from . import llama
from . import dummy
from . import phi
from . import mistral

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "opt": opt.OPTLM,
    "llama": llama.LLaMA,
    "dummy": dummy.DummyLM,
    "phi": phi.PHI,
    "mistral": mistral.MISTRAL
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
