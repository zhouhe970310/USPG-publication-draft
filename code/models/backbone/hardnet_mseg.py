import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """基础卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class HarDBlock(nn.Module):
    """HarDNet Block - Harmonic Dense Block"""
    def __init__(self, in_channels, growth_rate, n_layers):
        super(HarDBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.out_channels_list = []

        for i in range(n_layers):
            # 计算输入通道数（harmonic连接）
            if i == 0:
                in_ch = in_channels
            else:
                # Harmonic连接：只连接2^k步前的层
                in_ch = in_channels
                for j in range(i):
                    if (i - j) & i == 0:  # 2的幂次
                        in_ch += growth_rate

            self.layers.append(ConvLayer(in_ch, growth_rate, kernel_size=3))
            self.out_channels_list.append(in_ch)

        # 最终输出通道数
        self.out_channels = in_channels + growth_rate * n_layers

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.layers):
            # Harmonic连接
            if i == 0:
                out = layer(x)
            else:
                # 连接2^k步前的所有层
                connects = [x]
                for j in range(i):
                    if (i - j) & i == 0:
                        connects.append(outputs[j + 1])
                out = layer(torch.cat(connects, dim=1))
            outputs.append(out)

        return torch.cat(outputs, dim=1)


class HarDNet(nn.Module):
    """HarDNet Backbone"""
    def __init__(self, in_channels=3):
        super(HarDNet, self).__init__()

        # 第一层
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=2, padding=1)

        # HarDNet blocks
        self.block1 = HarDBlock(32, growth_rate=16, n_layers=4)
        self.down1 = nn.MaxPool2d(2, 2)

        self.block2 = HarDBlock(self.block1.out_channels, growth_rate=16, n_layers=4)
        self.down2 = nn.MaxPool2d(2, 2)

        self.block3 = HarDBlock(self.block2.out_channels, growth_rate=20, n_layers=8)
        self.down3 = nn.MaxPool2d(2, 2)

        self.block4 = HarDBlock(self.block3.out_channels, growth_rate=24, n_layers=8)
        self.down4 = nn.MaxPool2d(2, 2)

        self.block5 = HarDBlock(self.block4.out_channels, growth_rate=32, n_layers=4)

    def forward(self, x):
        # 编码路径
        x1 = self.conv1(x)  # 1/2

        x2 = self.block1(x1)  # 1/2
        x2_down = self.down1(x2)  # 1/4

        x3 = self.block2(x2_down)  # 1/4
        x3_down = self.down2(x3)  # 1/8

        x4 = self.block3(x3_down)  # 1/8
        x4_down = self.down3(x4)  # 1/16

        x5 = self.block4(x4_down)  # 1/16
        x5_down = self.down4(x5)  # 1/32

        x6 = self.block5(x5_down)  # 1/32

        return [x2, x3, x4, x5, x6]


class MSEGDecoder(nn.Module):
    """Multi-Scale Edge Guidance Decoder"""
    def __init__(self, encoder_channels):
        super(MSEGDecoder, self).__init__()

        # 解码器通道数
        decoder_channels = [256, 128, 64, 32]

        # 上采样和融合模块
        self.up5 = nn.Sequential(
            nn.Conv2d(encoder_channels[4], decoder_channels[0], 1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(encoder_channels[3] + decoder_channels[0], decoder_channels[1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(encoder_channels[2] + decoder_channels[1], decoder_channels[2], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(encoder_channels[1] + decoder_channels[2], decoder_channels[3], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True)
        )

        # 最终预测头
        # Parameter-free stochastic head used during training and MC-dropout inference.
        self.dropout = nn.Dropout2d(p=0.12)
        self.final_conv = nn.Conv2d(decoder_channels[3], 1, 1)

    def forward(self, features):
        """
        Args:
            features: 来自编码器的特征 [x2, x3, x4, x5, x6]
        Returns:
            pred: 分割预测 (B, 1, H, W)
        """
        x2, x3, x4, x5, x6 = features

        # 上采样路径
        d5 = self.up5(x6)  # 1/32
        d5 = F.interpolate(d5, size=x5.shape[2:], mode='bilinear', align_corners=False)

        d4 = torch.cat([d5, x5], dim=1)  # 1/16
        d4 = self.up4(d4)
        d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=False)

        d3 = torch.cat([d4, x4], dim=1)  # 1/8
        d3 = self.up3(d3)
        d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=False)

        d2 = torch.cat([d3, x3], dim=1)  # 1/4
        d2 = self.up2(d2)

        # 最终上采样到原始尺寸
        d2 = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=False)

        # 预测
        pred = self.final_conv(self.dropout(d2))

        return pred


class HarDNetMSEG(nn.Module):
    """HarDNet-MSEG完整模型"""
    def __init__(self, in_channels=3):
        super(HarDNetMSEG, self).__init__()

        self.encoder = HarDNet(in_channels)

        # 获取编码器输出通道数
        encoder_channels = [
            96,   # x2
            160,  # x3
            320,  # x4
            512,  # x5
            640   # x6
        ]

        self.decoder = MSEGDecoder(encoder_channels)

    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            pred: 分割预测 logits (B, 1, H, W)
        """
        features = self.encoder(x)
        pred = self.decoder(features)
        return pred


def test_hardnet_mseg():
    """测试HarDNet-MSEG模型"""
    model = HarDNetMSEG(in_channels=3)
    x = torch.randn(2, 3, 352, 352)

    print("Input shape:", x.shape)

    # 前向传播
    pred = model(x)
    print("Output shape:", pred.shape)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    return model


if __name__ == '__main__':
    test_hardnet_mseg()
