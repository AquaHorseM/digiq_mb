import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim=3584, base_channels=128, output_img_size=256):
        super().__init__()
        self.init_spatial  = output_img_size // 32  # e.g., 8
        self.init_channels = base_channels * 8      # e.g., 128*8=1024

        # 1. FC 映射到 (init_channels, init_spatial, init_spatial)
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_spatial * self.init_spatial)

        # 2. 设计更大的转置卷积核并行小卷积支路
        def deconv_block(in_ch, out_ch, large_k=7, small_k=3):
            pad_large = large_k // 2
            pad_small = small_k // 2
            return nn.ModuleDict({
                'deconv_large':      nn.ConvTranspose2d(in_ch, out_ch, kernel_size=large_k, stride=2, padding=pad_large, output_padding=large_k%2),
                'deconv_small':      nn.ConvTranspose2d(in_ch, out_ch, kernel_size=small_k, stride=2, padding=pad_small, output_padding=small_k%2),
                'bn':                nn.BatchNorm2d(out_ch),
                'extra_conv':        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                'extra_bn':          nn.BatchNorm2d(out_ch),
            })

        # 3. 定义各上采样阶段
        self.up1 = deconv_block(self.init_channels, base_channels*8, large_k=9, small_k=3)  # 8→16
        self.up2 = deconv_block(base_channels*8, base_channels*4, large_k=7, small_k=3)   # 16→32
        self.up3 = deconv_block(base_channels*4, base_channels*2, large_k=5, small_k=3)   # 32→64
        self.up4 = deconv_block(base_channels*2, base_channels  , large_k=3, small_k=3)   # 64→128

        # 4. 使用 PixelShuffle 进行最后一次上采样到 256
        self.conv_ps = nn.Conv2d(base_channels, 3 * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.final_activation = nn.Sigmoid()

        # 5. 每次上采样后的残差模块
        def make_resblock(channels):
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
            )
        self.res1 = make_resblock(base_channels*8)
        self.res2 = make_resblock(base_channels*4)
        self.res3 = make_resblock(base_channels*2)
        self.res4 = make_resblock(base_channels)

    def forward(self, z):
        # FC 映射并 reshape
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_spatial, self.init_spatial)

        # 上采样 1
        a1 = self.up1['deconv_large'](x)
        b1 = self.up1['deconv_small'](x)
        x1 = F.relu(self.up1['bn'](a1 + b1))
        x1 = F.relu(self.up1['extra_bn'](self.up1['extra_conv'](x1)))
        # 残差连接
        x1 = F.relu(self.res1(x1) + x1)

        # 上采样 2
        a2 = self.up2['deconv_large'](x1)
        b2 = self.up2['deconv_small'](x1)
        x2 = F.relu(self.up2['bn'](a2 + b2))
        x2 = F.relu(self.up2['extra_bn'](self.up2['extra_conv'](x2)))
        x2 = F.relu(self.res2(x2) + x2)

        # 上采样 3
        a3 = self.up3['deconv_large'](x2)
        b3 = self.up3['deconv_small'](x2)
        x3 = F.relu(self.up3['bn'](a3 + b3))
        x3 = F.relu(self.up3['extra_bn'](self.up3['extra_conv'](x3)))
        x3 = F.relu(self.res3(x3) + x3)

        # 上采样 4
        a4 = self.up4['deconv_large'](x3)
        b4 = self.up4['deconv_small'](x3)
        x4 = F.relu(self.up4['bn'](a4 + b4))
        x4 = F.relu(self.up4['extra_bn'](self.up4['extra_conv'](x4)))
        x4 = F.relu(self.res4(x4) + x4)

        # 最后一层 PixelShuffle 上采样
        x5 = self.conv_ps(x4)
        x5 = self.pixel_shuffle(x5)
        out = self.final_activation(x5)

        return out