lr=1e-4
lora_adapter_rank=64
lora_adapter_alpha=128
lora_adapter_trainable="q_proj,v_proj"
lora_adapter_dropout=0.05

pretrained_model=/root_path/DeepDR-LLM/Module1/llama-7b-weights
chinese_tokenizer_path=/root_path/DeepDR-LLM/Module1/llama-7b-weights
dataset_dir=/root_path/DeepDR-LLM/Module1/Minimum Dataset/train_set/
per_device_train_batch_size=8
gradient_accumulation_steps=8
max_seq_length=512
output_dir=/root_path/DeepDR-LLM/Module1/lora-adapter-weights
validation_file=/root_path/DeepDR-LLM/Module1/Minimum Dataset/valid_set.json
deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 1 LLM_train.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed 40 \
    --fp16 \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_adapter_rank} \
    --lora_alpha ${lora_adapter_alpha} \
    --trainable ${lora_adapter_trainable} \
    --lora_dropout ${lora_adapter_dropout} \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
    
