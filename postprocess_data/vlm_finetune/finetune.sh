conda activate llava

# user specified
llava_repo_dir=<path/to/LLaVA>
cache_dir=<path/to/.cache>
working_dir="" # should be the same as the working_dir in postprocess_data/get_more_actions/RUNME.sh

# do not modify these
llava_training_datapoints_path=$working_dir/datapoints-llava-contrastive.json
layover_image_dir=$working_dir/cursor_images
llava_save_dir=$working_dir/llava-v1.5-7b-sfted-pad

# if you want to specify GPUs to use: deepspeed --master_port=25678 --include localhost:6 <path_to_LLaVA_repo>/LLaVA/llava/train/train_mem.py \
# if you want to use lora: --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
deepspeed --master_port=25678 $llava_repo_dir/llava/train/train_mem.py \
    --deepspeed $llava_repo_dir/scripts/zero3.json \
    --cache_dir $cache_dir \
    --data_path $llava_training_datapoints_path \
    --image_folder $layover_image_dir \
    --output_dir $llava_save_dir \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --image_aspect_ratio pad \
    --bf16 True \
    --tf32 True \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --num_train_epochs 0.01 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
