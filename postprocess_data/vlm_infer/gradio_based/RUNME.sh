working_dir="" # should be the same as the working_dir in postprocess_data/get_more_actions/RUNME.sh
num_gpus=4
num_actions=16
num_processes_per_gpu=2 # each process takes 16G GPU memory

########

# # 1. launch LLaVA model with gradio
# eval "$(conda shell.bash hook)"
# conda activate llava
# rm llava_launched_places.out
# for gpu in $(seq 0 $((num_gpus - 1)))
# do
#     for i in $(seq 0 $((num_processes_per_gpu - 1)))
#     do
#         port=$((7860 + gpu * 4 + i))
#         CUDA_VISIBLE_DEVICES=$gpu nohup python -u host.py \
#         --model-path $working_dir/llava-v1.5-7b-sfted-pad \
#         --port $port \
#         >> llava_launched_places.out &
#     done
# done

# sleep 60 # wait for servers to launch

# # 2. get the URLs of the launched LLaVA models
# python aggregate_urls.py

# echo "now copy & paste the URLs to client.py line 19, comment section 1 and 2 out, uncomment section 3, and run this script again"

########

# 3. host the LLaVA model with gradio
python client.py \
  --input_data_path $working_dir/trajectories-nv2v-${num_actions}actions-qimages.pt \
  --output_data_path $working_dir/trajectories-nv2v-${num_actions}actions-qimages-llavarep.pt
