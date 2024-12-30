export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE=./data/
export PYTHONPATH=./

model=/mnt/publiccache/huggingface/Phi-3-mini-128k-instruct
model_name=phi3

for task in toolalpacaphi3 triviaqaphi3 apibankphi3 metamathqaphi3 codealpacaphi3 alpacaphi3
do
python main.py --model phi --model_args pretrained=${model} --tasks ${task} --head_importance_calc --save_importance_path logs/head_importance/${model_name}/0shot_${task}.pkl --num_fewshot 0
done
















































# model=/netcache/yphao/model/apibank/apibank_amount/checkpoint-500
# model_name=checkpoint-500

# # Attention Head Importance Scores
# for task in triviaqa apibank_level_1 gsm8k code align
# do
# python main.py --model llama --model_args pretrained=${model} --tasks ${task} --head_importance_calc --save_importance_path logs/head_importance/${model_name}/0shot_${task}.pkl --num_fewshot 0
# python scripts/plotting/heatmap_mlp.py --saved_head_importance_path logs/head_importance/${model_name}/0shot_${task}_mlp.pkl --dataset ${task} --save_plot_path logs/head_importance/${model_name}/0shot_${task}_mlp.pdf --shot 0-shot
# python scripts/plotting/heatmap_att.py --saved_head_importance_path logs/head_importance/${model_name}/0shot_${task}_att.pkl --dataset ${task} --save_plot_path logs/head_importance/${model_name}/0shot_${task}_att.pdf --shot 0-shot


# # plot
# python scripts/plotting/heatmap.py --saved_head_importance_path logs/head_importance/${model_name}/0shot_${task}.pkl --dataset ${task} --save_plot_path logs/head_importance/${model_name}/0shot_${task}_att_head.pdf --shot 0-shot
# done
# plot
# python scripts/plotting/heatmap.py --saved_head_importance_path logs/head_importance/${model_name}/0shot_${task}.pkl --dataset ${task} --save_plot_path logs/head_importance/${model_name}/0shot_${task}.pdf --shot 0-shot

# pruning fc
# for task in gsm8k code align
# do 
# for layer in {0..31}
# do 
#     python main.py --model llama --model_args pretrained=${model},mask_fc=${layer} --tasks ${task} --output_path results/${model_name}/0shot_fc_pruning/${task}/0shot_fc_${layer}.txt --batch_size 2 --num_fewshot 0
# done
# for task in gsm8k code align
# do 
# python main.py --model llama --model_args pretrained=${model},mask_fc=0/1/2/3/4/5/6/7 --tasks ${task} --output_path results/${model_name}/0shot_fc_pruning/${task}/0shot_fc_p1.txt --batch_size 2 --num_fewshot 0
# python main.py --model llama --model_args pretrained=${model},mask_fc=8/9/10/11/12/13/14/15 --tasks ${task} --output_path results/${model_name}/0shot_fc_pruning/${task}/0shot_fc_p2.txt --batch_size 2 --num_fewshot 0
# python main.py --model llama --model_args pretrained=${model},mask_fc=16/17/18/19/20/21/22/23 --tasks ${task} --output_path results/${model_name}/0shot_fc_pruning/${task}/0shot_fc_p3.txt --batch_size 2 --num_fewshot 0
# python main.py --model llama --model_args pretrained=${model},mask_fc=24/25/26/27/28/29/30/31 --tasks ${task} --output_path results/${model_name}/0shot_fc_pruning/${task}/0shot_fc_p4.txt --batch_size 2 --num_fewshot 0
# done

# counter=0  
# for (( ; counter<=90; counter+=10 ))  
# do  
#     python main.py --model llama --model_args pretrained=${model},mask_heads=1,head_importance_path=logs/head_importance/${model_name}/0shot_${task}.pkl,head_percent_mask=${counter} --tasks ${task} --output_path results/${model_name}/${task}/0shot_${counter}_percent.txt --batch_size 2 --num_fewshot 0
# done
# # plot
# python scripts/plotting/fc_importance_single.py --dataset ${task} --results_path results/${model_name}/0shot_fc_pruning/ --base_results_path results/${model_name}/ --shot 0-shot --save_plot_path paper_plots/fc_importance/0-shot.pdf --dump_fc_importance --dump_fc_importance_path logs/fc_knocking_importance/


# python main.py --model opt --model_args pretrained=facebook/opt-66b,model_cache_dir=opt66b_checkpoints,tokenizer_cache_dir=opt66b_tokenizer,mask_iterative_fc=1,fc_importance_path=logs/fc_knocking_importance/1shot_piqa.pkl,fc_percent_mask=30,mask_heads=1,head_importance_path=logs/head_importance/opt66b/1shot_piqa.pkl,head_percent_mask=20 --tasks piqa --output_path results/${model_name}/${task}/0shot_30_fc_20_head_percent.txt --batch_size 2 --num_fewshot 0





# model=/mnt/publiccache/yphao/model/apibank/apibank_llama2_chat_llama_full_sft_checkpoint
# model_name=apibank_llama2

# Attention Head Importance Scores
# python main.py --model llama --model_args pretrained=${model} --tasks ${task} --head_importance_calc --save_importance_path logs/head_importance/${model_name}/0shot_${task}.pkl --num_fewshot 0
# plot
# python scripts/plotting/heatmap.py --saved_head_importance_path logs/head_importance/${model_name}/0shot_${task}.pkl --dataset ${task} --save_plot_path logs/head_importance/${model_name}/0shot_${task}.pdf --shot 0-shot

# for layer in {0..31}
# do 
#     python main.py --model llama --model_args pretrained=${model},mask_fc=${layer} --tasks ${task} --output_path results/${model_name}/0shot_fc_pruning/${task}/0shot_fc_${layer}.txt --batch_size 2 --num_fewshot 0
# done

# counter=0
# for (( ; counter<=90; counter+=10 ))  
# do  
#     python main.py --model llama --model_args pretrained=${model},mask_heads=1,head_importance_path=logs/head_importance/${model_name}/0shot_${task}.pkl,head_percent_mask=${counter} --tasks ${task} --output_path results/${model_name}/${task}/0shot_${counter}_percent.txt --batch_size 2 --num_fewshot 0
# done
# # plot
# python scripts/plotting/fc_importance_single.py --dataset ${task} --results_path results/${model_name}/0shot_fc_pruning/ --base_results_path results/${model_name}/ --shot 0-shot --save_plot_path paper_plots/fc_importance/0-shot.pdf --dump_fc_importance --dump_fc_importance_path logs/fc_knocking_importance/
