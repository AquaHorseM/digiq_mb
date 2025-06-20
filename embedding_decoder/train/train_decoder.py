import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split

from data_utils import EmbeddingImageDataset
from embedding_decoder.train.model import Decoder

def reconstruction_loss(recon_img, real_img, loss_type='mse'):
    """
    计算生成图像与真实图像之间的重构损失。
    Args:
        recon_img: Tensor, (B,3,H,W)，Decoder 的输出
        real_img:  Tensor, (B,3,H,W)，真实图像
        loss_type: 'mse' 或 'bce'，默认为 'mse'
    Returns:
        loss: 标量 Tensor，批量总损失（sum over all像素）
    """
    if loss_type == 'mse':
        # 适用于 Tanh 输出（范围 [-1,1]）或一般回归
        loss = F.mse_loss(recon_img, real_img, reduction='sum')
    elif loss_type == 'bce':
        # 适用于 Sigmoid 输出（范围 [0,1]）
        loss = F.binary_cross_entropy(recon_img, real_img, reduction='sum')
    else:
        raise ValueError("loss_type 必须是 'mse' 或 'bce'")
    return loss

def train(epoch, model, dataloader, optimizer, device, beta):
    model.train()
    train_loss = 0.0

    loop = tqdm(dataloader, desc=f"Epoch [{epoch}]")
    for batch_idx, data in enumerate(loop):
        # data: (B,3,img_size,img_size)
        embeds, images = data[0], data[1]
        embeds = embeds.to(device)
        images = images.to(device)
        optimizer.zero_grad()

        recon_batch = model(embeds)
        loss = reconstruction_loss(recon_batch, images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        loop.set_postfix({
            'loss': loss.item() / embeds.size(0)
        })

    avg_loss = train_loss / len(dataloader.dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss

def validate(epoch, model, dataloader, device, beta):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            embeds, images = data[0], data[1]
            embeds = embeds.to(device)
            images = images.to(device)
            recon_batch = model(embeds)
            loss = reconstruction_loss(recon_batch, images)
            val_loss += loss.item()
    avg_loss = val_loss / len(dataloader.dataset)
    print(f"----> Validation: Epoch {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 随机种子
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # 1. 定义图像预处理
    def pad_to_square(img: Image.Image, fill=0):
        """
        将任意长宽的 PIL.Image pad 成正方形：
        - 在短的一边两侧均匀填充，使得最终宽 == 高 == max(原宽, 原高)
        - fill 可以是单个数（灰度或黑白图像）或三元组 (R,G,B)
        """
        w, h = img.size
        max_wh = max(w, h)
        pad_left   = (max_wh - w) // 2
        pad_top    = (max_wh - h) // 2
        pad_right  = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill)

    # 构造 transform
    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_square(img, fill=0)),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),          # 转为 [0,1] 的 Tensor, 形状 [3,img_size,img_size]
    ])

    # 2. 实例化 Dataset
    dataset = EmbeddingImageDataset(
        root_dir=args.data_root,
        embedding_subdir="embeddings",
        image_subdir="images",
        transform=transform
    )  # :contentReference[oaicite:26]{index=26}

    # 3. 随机划分：80% 作为训练集，20% 作为验证集
    N = len(dataset)               # 数据集总大小 :contentReference[oaicite:17]{index=17}
    n_train = int(0.8 * N)
    n_val   = N - n_train

    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], 
        generator=torch.Generator().manual_seed(42)  # 保证可复现
    )  # :contentReference[oaicite:18]{index=18}

    # 4. 构造 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,      # 对于 Subset，shuffle=True 会重新 shuffle 索引列表
        num_workers=4,
        pin_memory=True
    )  # :contentReference[oaicite:19]{index=19}

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )  # :contentReference[oaicite:20]{index=20}

    # 模型、优化器与学习率调度
    model = Decoder(latent_dim=args.latent_dim, base_channels=args.base_channels, output_img_size=args.img_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # （可选）如果使用多卡：
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    # 继续训练 / 断点恢复（可选）
    start_epoch = 1
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {args.resume_checkpoint}, starting at epoch {start_epoch}")

    # 日志目录
    os.makedirs(args.save_dir, exist_ok=True)

    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, device, args.beta)
        val_loss = validate(epoch, model, val_loader, device, args.beta)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, f"best_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print("Saved best model to", save_path)

        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            print("Saved checkpoint to", checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep VAE Training (Latent Dim = 2048)")
    parser.add_argument("--data_root", type=str, default="data", help="数据集根目录，包含 train/ 和 val/ 子文件夹")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型与日志保存目录")
    parser.add_argument("--img_size", type=int, default=512, help="输入图像尺寸（假设正方形）")
    parser.add_argument("--batch_size", type=int, default=32, help="训练 Batch 大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--latent_dim", type=int, default=3584, help="VAE latent 维度")
    parser.add_argument("--base_channels", type=int, default=64, help="基础通道数，控制网络宽度")
    parser.add_argument("--beta", type=float, default=1.0, help="KL 散度权重")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="继续训练的 checkpoint 路径")
    parser.add_argument("--save_every", type=int, default=10, help="每 N 个 epoch 保存一次 checkpoint")
    args = parser.parse_args()

    main(args)
