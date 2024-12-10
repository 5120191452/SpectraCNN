import torch
import torch.nn as nn
from CBAM import CBAMBlock  # 假设CBAMBlock已经定义


class DepthwiseSeparableResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(DepthwiseSeparableResidualBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果输入和输出的通道数不同，我们需要一个卷积层来匹配维度
        self.match_channels = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        # Depthwise separable convolution
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 如果输入和输出的通道数不同，使用1x1卷积匹配维度
        if self.match_channels is not None:
            residual = self.match_channels(residual)

        # Residual connection
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual  # Add skip connection
        return self.relu(x)


class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )

        # 第一个卷积块
        self.ds_res_block = DepthwiseSeparableResidualBlock(16, 32, padding=1)

        # 注意力模块
        self.cbam = CBAMBlock(32)

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 展平层和全连接层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 2 * 2, num_classes + 1)  # 特征图在最后一个池化后的大小为2x2

    def forward(self, x):
        x = self.conv1(x)
        x = self.ds_res_block(x)
        x = self.cbam(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
def count_parameters(model):
    """计算模型每一层的参数量"""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params
def print_model_parameters(model):
    """打印模型每一层的参数量"""
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} \t Parameters: {param.numel()}")
    print(f"Total parameters: {count_parameters(model)}")

if __name__ == "__main__":
    model = MyCNN(num_classes=9)
    # Print the model summary
    print(model)
    # 计算参数量
    print_model_parameters(model)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total number of parameters is: ", total_params)