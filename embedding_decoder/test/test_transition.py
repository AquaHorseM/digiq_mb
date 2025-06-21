import torch

from digiq.models.encoder import ActionEncoder
from digiq.models.transition_model import  MLPTransition, TransformerTransition

from embedding_decoder.train.model import Decoder

from PIL import Image
import numpy as np
import re

def load_decoder(checkpoint_path, latent_dim=3584, base_channels=128, output_img_size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = Decoder(latent_dim=latent_dim,
                              base_channels=base_channels,
                              output_img_size=output_img_size)
    decoder.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()
    return decoder

def decode_embedding_to_image(decoder, z, output_path):
    """
    将给定的 embedding z 解码为图像并保存到 output_path。
    - decoder: 已加载的 ImprovedDecoder 实例（eval 模式）。
    - z: Tensor，形状 [batch_size, latent_dim]，在 decoder 所在的 device 上。
    - output_path: 保存路径（仅保存第 0 个样本）。
    返回 PIL.Image 对象。
    """
    device = next(decoder.parameters()).device
    z = z.to(device)
    with torch.no_grad():
        output_tensor = decoder(z)
    
    # 仅处理第 0 个样本
    for idx, output in enumerate(output_tensor):
        img_tensor = output.cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(output_path)

def parse_obs(observation):
    if type(observation) == str:
        observation = [observation]
    
    # obs example: Previous Actions: Goal: Go to newegg.com</s>
    previous_actions = []
    goals = []
    for obs in observation:
        previous_action_match = re.search(r'Previous Actions: (.*?)Goal:', obs)
        goal_match = re.search(r'Goal: (.*?)</s>', obs)
        
        # Map None to an empty string if no match is found
        previous_actions.append(previous_action_match.group(1) if previous_action_match else "")
        goals.append(goal_match.group(1) if goal_match else "")

    return previous_actions, goals

if __name__ == "__main__":
    # 配置参数
    decoder_path = "/data/mqj/models/vae_512/best_epoch_47.pth"
    data_path = "/data/mqj/datasets/rl/general-ft.pt"
    transition_path = "/data/mqj/models/full_transition/digiq_TransitionModel_tf512-2.pth"
    output_dir = "embedding_decoder/test/transition_result"
    sampled_idx = 512
    latent_dim = 3584
    base_channels = 64
    output_img_size = 512

    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = load_decoder(decoder_path,
                           latent_dim=latent_dim,
                           base_channels=base_channels,
                           output_img_size=output_img_size,
                           device=device)
    transition = TransformerTransition(
        state_dim=3584, action_dim=1536, device=device
    )
    transition.load_state_dict(torch.load(transition_path, map_location=device))
    action_encoder = ActionEncoder(backbone='roberta-base', cache_dir=None, device=device)

    data = torch.load(data_path, weights_only=False)
    step = data[sampled_idx]

    state = torch.from_numpy(step['s_rep']).unsqueeze(0).to(device)   
    action = action_encoder(step['action'])

    transition.eval()
    with torch.no_grad():
        pred_next_state = transition.forward(state, action)
    next_state = torch.from_numpy(step['next_s_rep']).unsqueeze(0).to(device)

    # 3. 解码并保存图像
    output_path = f"{output_dir}/{step['image_path'].split('/')[-2]}-{step['image_path'].split('/')[-1]}"
    decode_embedding_to_image(decoder, pred_next_state, output_path=output_path)
    print(f"next_state saved in {output_path}")
