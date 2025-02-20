working_dir="" # should be the same as the working_dir in postprocess_data/get_more_actions/RUNME.sh
num_gpus=4
num_actions=16 # keep same as num_actions_total in get_more_actions/RUNME.sh
# make sure (num_trajs // num_gpus) % num_finegrained_slices == 0
num_finegrained_slices=2

eval "$(conda shell.bash hook)"
cd vllm_based
conda activate llava

# 1. save LLaVA state dict from liuhaotian's repo
mkdir $working_dir/llava-v1.5-7b-sfted-pad-state-dict
python -u save_state_dict.py \
  --input_model_path $working_dir/llava-v1.5-7b-sfted-pad \
  --output_state_dict_path $working_dir/llava-v1.5-7b-sfted-pad-state-dict/model_state_dict.bin

# 2. convert state dict to HF format
python convert_weights.py \
    --text_model_id lmsys/vicuna-7b-v1.5 \
    --vision_model_id openai/clip-vit-large-patch14-336 \
    --output_hub_path $working_dir/llava-v1.5-7b-sfted-pad-hf-converted \
    --old_state_dict_id $working_dir/llava-v1.5-7b-sfted-pad-state-dict

# 3. launch vllm
conda activate vllm
rm -f batch_launch_vllm_get_rep.out
llavarep_path_base=$working_dir/trajectories-nv2v-${num_actions}actions-qimages-llavarep
rm -f $llavarep_path_base-slice*

echo "Launching VLLM inference..."

for i in $(seq 0 $((num_gpus - 1)));
do
    slice=$i
    CUDA_VISIBLE_DEVICES=$i nohup python -u vllm_infer_batched.py \
        --slice-id $slice \
        --num-slices $num_gpus \
        --num-actions $num_actions \
        --gpu-memory-utilization 0.90 \
        --model-path $working_dir/llava-v1.5-7b-sfted-pad-hf-converted \
        --input-data-path $working_dir/trajectories-nv2v-${num_actions}actions-qimages.pt \
        --output-data-path $llavarep_path_base \
        --num-finegrained-slices $num_finegrained_slices \
        >> batch_launch_vllm_get_rep.out &
done

python -u merge_slices.py \
  --num_slices $num_gpus \
  --llavarep_path_base $llavarep_path_base

sleep 86400
