import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class EmbeddingImageDataset(Dataset):
    """
    自定义 Dataset：返回 (embedding, image) 对。

    数据目录结构（示例）：
      dataset_root/
      ├── embeddings/
      │   ├── img0001.npy
      │   ├── img0002.npy
      │   └── ...
      └── images/
          ├── img0001.png
          ├── img0002.png
          └── ...

    注意：embedding 文件名（不含 .npy）必须和图像文件名（不含扩展名）一致。
    """

    def __init__(self, root_dir, embedding_subdir="embeddings", image_subdir="images", transform=None):
        """
        Args:
            root_dir (str): 数据集根目录路径（如 'dataset_root/'）。
            embedding_subdir (str): 嵌入文件夹名称，默认 'embeddings'。
            image_subdir (str): 图像文件夹名称，默认 'images'。
            transform (torchvision.transforms, optional): 图像预处理变换，返回 [C,H,W] Tensor。
        """
        self.root_dir = root_dir  # e.g., "dataset_root/"
        self.embed_dir = os.path.join(root_dir, embedding_subdir)  # e.g., "dataset_root/embeddings"
        self.image_dir = os.path.join(root_dir, image_subdir)      # e.g., "dataset_root/images"
        self.transform = transform  # 图像变换

        # 检查目录是否存在
        if not os.path.isdir(self.embed_dir):
            raise ValueError(f"Embedding 目录不存在：{self.embed_dir}")  # :contentReference[oaicite:9]{index=9}
        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Image 目录不存在：{self.image_dir}")    # :contentReference[oaicite:10]{index=10}

        # 获取所有 embedding 文件列表，保留文件名前缀（去掉 .npy）
        all_embed_files = [
            fname[:-4]  # 去掉 ".npy" 后缀
            for fname in os.listdir(self.embed_dir)
            if fname.endswith(".npy")
        ]  # :contentReference[oaicite:11]{index=11}

        # 过滤：仅保留在 images/ 中存在对应图像的 embedding 标识
        self.sample_names = []
        for name in sorted(all_embed_files):  # 排序以保证顺序确定性
            # 支持多种图像扩展名，如 .png 或 .jpg
            img_path_png = os.path.join(self.image_dir, name + ".png")
            img_path_jpg = os.path.join(self.image_dir, name + ".jpg")
            img_path_jpeg = os.path.join(self.image_dir, name + ".jpeg")
            if os.path.isfile(img_path_png):
                self.sample_names.append((name, img_path_png))
            elif os.path.isfile(img_path_jpg):
                self.sample_names.append((name, img_path_jpg))
            elif os.path.isfile(img_path_jpeg):
                self.sample_names.append((name, img_path_jpeg))
            else:
                # 如果对应图像文件缺失，跳过该 embedding
                # 可根据需要打印警告
                # print(f"警告：未找到对应图像，跳过 embedding {name}")
                continue  # :contentReference[oaicite:12]{index=12}

        if len(self.sample_names) == 0:
            raise ValueError(f"未在 '{self.embed_dir}' 和 '{self.image_dir}' 中找到任何有效的配对文件")  # :contentReference[oaicite:13]{index=13}

    def __len__(self):
        """
        返回数据集中样本的数量（即有效的 embedding-image 对数）。
        """
        return len(self.sample_names)  # :contentReference[oaicite:14]{index=14}

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一个 (embedding, image) 对：
          - embedding: Tensor, 形状 [2048]
          - image:     Tensor, 形状 [3, H, W] (经过 transform 后)

        步骤：
          1. 从 self.sample_names[idx] 拿到 (name, image_path)。
          2. 拼接 embedding 文件路径：{embed_dir}/{name}.npy。
          3. 用 numpy.load 读取 .npy 文件，将得到 shape=[2048] 的 np.ndarray。
          4. 转为 float32，并用 torch.from_numpy -> Tensor。
          5. 读取图像：PIL.Image.open(image_path)。应用 self.transform -> Tensor。
          6. 返回 (embedding_tensor, image_tensor)。
        """
        name, img_path = self.sample_names[idx]

        # 1. 加载 embedding
        embed_path = os.path.join(self.embed_dir, name + ".npy")
        try:
            embedding_np = np.load(embed_path)  # np.ndarray, shape=[2048]  :contentReference[oaicite:15]{index=15}
        except Exception as e:
            raise RuntimeError(f"无法加载 embedding 文件 {embed_path}: {e}")  # :contentReference[oaicite:16]{index=16}

        # 确保数据类型为 float32
        if not isinstance(embedding_np, np.ndarray):
            raise TypeError(f"numpy.load 返回类型不是 np.ndarray，而是 {type(embedding_np)}")
        embedding_np = embedding_np.astype(np.float32)  # :contentReference[oaicite:17]{index=17}

        # 转为 PyTorch Tensor
        embedding_tensor = torch.from_numpy(embedding_np)  # shape=[2048]  :contentReference[oaicite:18]{index=18}

        # 2. 加载图像
        image = Image.open(img_path).convert("RGB")  # 统一为 RGB 模式  :contentReference[oaicite:19]{index=19}
        if self.transform is not None:
            image_tensor = self.transform(image)  # e.g., transforms.ToTensor() -> [3,H,W], 范围 [0,1]  :contentReference[oaicite:20]{index=20}
        else:
            # 如果没有提供 transform，手动将 PIL 转 Tensor，且归一化到 [0,1]
            image_tensor = transforms.ToTensor()(image)  # :contentReference[oaicite:21]{index=21}

        return embedding_tensor, image_tensor
