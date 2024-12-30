export CUDA_VISIBLE_DEVICES=0

export model="/mnt/publiccache/yphao/model/apibank/apibank_mix_llama3_instruct_CITI_stage_3"
export base_model="/mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct"
export moelora_allocation="/mnt/publiccache/yphao/model/moelora_allocation/llama3_moe_allocation_top_20_percent.json"

python predict_moe_llama3.py --model_path ${model} --base_model ${base_model} --moelora_allocation ${moelora_allocation} 

