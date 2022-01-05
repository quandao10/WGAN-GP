import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv2D_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        self.conv2D_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv2D_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv2D_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.conv2D_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2)
        self.conv2D_6 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv2D_1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2D_2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2D_3(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2D_4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2D_5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2D_6(x)
        return x


# model = Critic().to("cuda")
# x = model(torch.randn(2, 3, 128, 128).to("cuda"))
