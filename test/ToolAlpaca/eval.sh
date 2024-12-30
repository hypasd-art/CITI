export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY=  # set the OpenAI API key

python testing_llama3.py -api ./data/eval_real_authentication.json -out ./pred/ --model_path /mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct

python testing_llama3_moe.py -api ./data/eval_real_authentication.json -out ./pred/ --model_path /mnt/publiccache/yphao/model/toolalpaca/toolalpaca_mix_llama3_instruct_CITI_stage_3 --base_model /mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct --moelora_allocation /mnt/publiccache/yphao/model/toolalpaca_moelora_allocation/llama3_moe_allocation_top_20_percent.json


python evaluation.py -api ./pred/Meta-Llama-3-8B-Instruct.json -out ./evaluation/Meta-Llama-3-8B-Instruct.json

