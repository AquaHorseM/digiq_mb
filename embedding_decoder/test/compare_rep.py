import torch
from embedding_decoder.train.model import Decoder
from PIL import Image
import numpy as np

def load_decoder(checkpoint_path, latent_dim=3584, base_channels=128, output_img_size=256, device=None):
    """
    加载预训练的 ImprovedDecoder。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = ImprovedDecoder(latent_dim=latent_dim,
                              base_channels=base_channels,
                              output_img_size=output_img_size)
    decoder.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()
    return decoder

def decode_embedding_to_image(decoder, z, output_path="."):
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
        img_pil.save(f"{output_path}/{idx}.png")

if __name__ == "__main__":
    # 配置参数
    checkpoint_path = "/data/mqj/models/vae_512/best_epoch_8.pth"
    data_path1 = "/data/mqj/datasets/rl/general.pt"
    data_path2 = "/data/mqj/datasets/rl/general-ft.pt"
    output_dir = "compared_result"
    sampled_idx = 1

    latent_dim = 3584
    base_channels = 64
    output_img_size = 512

    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = load_decoder(checkpoint_path,
                           latent_dim=latent_dim,
                           base_channels=base_channels,
                           output_img_size=output_img_size,
                           device=device)

    # 2. 获取 embedding（或从其它模块获取）
    data1 = torch.load(data_path1, weights_only=False)
    state1 = torch.from_numpy(data1[sampled_idx]['s_rep']).to(device)
    data2 = torch.load(data_path2, weights_only=False)
    state2 = torch.from_numpy(data2[sampled_idx]['s_rep']).to(device)
    z = torch.stack([state1, state2])
    

    # 3. 解码并保存图像
    decode_embedding_to_image(decoder, z, output_path=output_dir)
    print(f"图像已保存到 {output_dir}")
