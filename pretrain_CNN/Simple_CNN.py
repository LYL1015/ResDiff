import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, scale_factor=4):
        super(SimpleCNN, self).__init__()

        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3 * scale_factor ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x + x_up

if __name__ == '__main__':
    x=torch.randn(7,3,32,32)
    net=SimpleCNN()
    y=net(x)