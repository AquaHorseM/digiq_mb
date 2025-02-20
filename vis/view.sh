rm debug.log

echo "Running visualization script..."

agent_path="" # this will just be the downloaded HF agent directory, named `aitw-webshop-digiq-agent` or `aitw-general-digiq-agent`
data_path="" # this will be the path to downloaded trajectory data, named `general-1008trajs.pt` or `webshop-1296trajs.pt`
image_path="" # this will be the path to unzipped images, named `general-images` or `webshop-images`
autoui_path="" # this will be the path to the downloaded autoui directory, named `Auto-UI-Base`
hf_cache_path=""

advantage_estimation="bellman" # keep this "bellman" for now 
learn_metric="regression" # keep this "regression" for now
click_icon_path="../assets/click.png" # this is the path to the click icon, you can use the default one "./assets/click.png"
num_actions=20 # this is the number of actions you want to visualize, keep it 20 for now

CUDA_VISIBLE_DEVICES=0 nohup python -u view_trajs.py \
    --agent_path $agent_path \
    --data_path $data_path \
    --image_path $image_path \
    --autoui_path $autoui_path \
    --hf_cache_path $hf_cache_path \
    --click_icon_path $click_icon_path \
    --advantage_estimation $advantage_estimation \
    --learn_metric $learn_metric \
    >> debug.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u view_actions.py \
    --agent_path $agent_path \
    --data_path $data_path \
    --image_path $image_path \
    --autoui_path $autoui_path \
    --hf_cache_path $hf_cache_path \
    --click_icon_path $click_icon_path \
    --num_actions $num_actions \
    --advantage_estimation $advantage_estimation \
    --learn_metric $learn_metric \
    >> debug.log 2>&1 &
    