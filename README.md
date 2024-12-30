# CITI: Enhancing Tool Utilizing Ability in Large Language Models without Sacrificing General Performance 

### Abstract

Tool learning enables Large Language Models (LLMs) to interact with the external environment by invoking tools, enriching the accuracy and capability scope of LLMs. However, previous works predominantly focus on improving the model's tool-utilizing accuracy and the ability to generalize to new, unseen tools, excessively forcing LLMs to adjust specific tool-invoking pattern without considering the harm to the model's general performance. This deviates from the actual applications and original intention of integrating tools to enhance the model. To tackle this problem, we dissect the capability trade-offs by examining the hidden representation changes and the gradient-based importance score of the model's components. Based on the analysis result, we propose a Component Importance-based Tool-utilizing ability Injection method (CITI). According to the gradient-based importance score of different components, it alleviates the capability conflicts caused by the fine-tuning process by applying distinct training strategies to different components. CITI applies Mixture-Of-LoRA (MOLoRA) for important components. Meanwhile, it fine-tunes the parameters of few components deemed less important in the backbone of the LLM, while keeping other parameters frozen. CITI can effectively enhance the model's tool-utilizing capability without excessively compromising its general performance. Experimental results demonstrate that our approach achieves outstanding performance across a range of evaluation metrics.

### Importance score computing

We calculate the gradient-base importance of the components in LLMs. In our experiments, we calculate the importance of components in different tasks, which represents one ability of LLM. The tasks are as follows: coding, mathematics, factual knowledge, instruction following and tool-utilizing. The datasets used to compute importance score are same as the datasets mentioned in training set. For each dataset, we randomly sample 3000 examples to compute the importance score. 


The code for importance computing is modified from llm-interpret(https://github.com/amazon-science/llm-interpret).

If you want to get the important components of LLaMA-3 on dataset API-Bank, you can run the following command:
```bash
cd ./lm_evaluation_harness
bash run_llama3.sh
python get_important_components_llama3.py
```


### Hidden representation analysis

The code about the hidden representation is in folder **hidden_representation_analysis**.



### Training Set Construction

To train CITI, We random sample 5000 instructions from datasets in the field of coding (CodeAlpaca-20K)(https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k), mathematics (MetaMathQA)(https://huggingface.co/datasets/meta-math/MetaMathQA), factual knowledge (TriviaQA)(https://huggingface.co/datasets/mandarjoshi/trivia_qa) and instruction following(https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) respectively, and mix them with the original tool training data in API-Bank. For ToolAlpaca, we sample 3000 training data from each datasets respectively.

To get the training and the testing data, please refer to code in folder **dataset**.

### Training

Our models are trained by LLaMA-Factory(https://github.com/hiyouga/LLaMA-Factory). And we modify the dialogue template of training data to make it fit the template required by different models. The code in folder **CITI/LLaMA-Factory/src/llamafactory/model/modeling_llama_moe** is copied and modified from code in https://github.com/Ablustrund/LoRAMoE and https://github.com/huggingface/transformers.

if you want to train the model in CITI method using API-Bank dataset, you can run the following command

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install -r requirements_our.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_apibank/llama3_CITI_sft_mix_stage_1.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_apibank/llama3_CITI_sft_mix_stage_2.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_apibank/llama3_CITI_sft_mix_stage_3.yaml
```



### Evaluation

Before evaluate the model on API-Bank and ToolAlpaca
```bash
cp CITI/LLaMA-Factory/src/llamafactory/model/modeling_llama_moe CITI/test/api_bank
cp CITI/LLaMA-Factory/src/llamafactory/model/modeling_llama_moe CITI/test/ToolAlpaca
```

The evaluation code is copied and modified from paper API-Bank(https://arxiv.org/abs/2304.08244) and ToolAlpaca(https://arxiv.org/abs/2306.05301).

To test the model on dataset API-Bank, please refer to the scripts in folder **CITI/test/api_bank/scripts**
To test the model on dataset ToolAlpaca, please refer to the script **CITI/test/ToolAlpaca/eval.sh**

The evaluation of general abilities follow the setting from OpenCompass(https://github.com/open-compass/opencompass) and MT-Bench(https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). 
