# this script is a working example for you to start with

eval "$(conda shell.bash hook)"
conda activate digiq

# all post-processing happens in $working_dir. Don't include "/" in the end
working_dir=""
click_icon_path=../../assets/click.png
autoui_base_path=""
num_actions_total=16
num_gpus=4

# 1. generate images with smaller resolution for LLaVA training (don't include "/" in the end)
original_image_dir=$working_dir/images
small_image_dir=$working_dir/smaller_images
python -u generate_smaller_images.py \
    --input_image_dir $original_image_dir \
    --output_image_dir $small_image_dir

# 2. redirect image paths in the trajectories to $working_dir/images
python -u redirect_image_paths.py \
    --input_data_path $working_dir/trajectories_eval.pt \
    --image_path $original_image_dir \
    --output_data_path $working_dir/trajectories_eval.pt

# 3. format collected data
# make sure your image_path matches the actual path before running this!
collected_data_path=$working_dir/trajectories_eval.pt
formatted_data_path=$working_dir/trajectories-nv2v.pt
python -u nv2v.py \
    --input_data_path $collected_data_path \
    --output_data_path $formatted_data_path

# 4. collect more actions (num_gpus should satisfy len(trajs) % num_gpus == 0)
more_actions_data_path=$working_dir/trajectories-nv2v-${num_actions_total}actions.pt
python -u get_more_actions_batched.py \
    --policy_lm $autoui_base_path \
    --num_actions $num_actions_total \
    --num_gpus $num_gpus \
    --input_data_path $formatted_data_path \
    --output_data_path $more_actions_data_path

# 5. overlay cursors onto image observations (num_slices should satisfy len(trajs) % num_slices == 0, and should be approaximate same as number of CPUs on your machine)
layover_image_dir=$working_dir/cursor_images
layover_image_data_path=$working_dir/trajectories-nv2v-${num_actions_total}actions-qimages.pt
mkdir $layover_image_dir
python -u generate_more_click_images_batched.py \
    --click_icon_path $click_icon_path \
    --num_slices 16 \
    --input_data_path $more_actions_data_path \
    --layovered_image_path $layover_image_dir \
    --output_data_path $layover_image_data_path

# 6. construct data that will later be used for contrastive training LLaVA
llava_training_datapoints_path=$working_dir/datapoints-llava-contrastive.json
python -u construct_sft_data.py \
    --input_data_path $layover_image_data_path \
    --output_contrastive_data_path $llava_training_datapoints_path
