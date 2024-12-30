export CUDA_VISIBLE_DEVICES=0

python predict_llama3.py --model_path /mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct


python eval.py --model Meta-Llama-3-8B-Instruct
