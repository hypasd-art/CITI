export CUDA_VISIBLE_DEVICES=0

export model="Meta-Llama-3-8B-Instruct" 

python eval.py --model ${model}

python eval.py --model Meta-Llama-3-8B-Instruct
python eval.py --model apibank_mix_llama3_instruct_CITI_stage_3