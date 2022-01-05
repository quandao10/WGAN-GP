import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2D_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.conv2D_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=128)
        self.conv2D_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(num_features=256)
        self.conv2D_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(num_features=512)
        self.conv2D_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=0)
        self.batch_norm_5 = nn.BatchNorm2d(num_features=1024)
        self.conv2D_6 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv2D_1(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.batch_norm_1(x)
        x = self.conv2D_2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.batch_norm_2(x)
        x = self.conv2D_3(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.batch_norm_3(x)
        x = self.conv2D_4(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.batch_norm_4(x)
        x = self.conv2D_5(x)
        x = self.batch_norm_5(x)
        x = self.conv2D_6(x)
        x = torch.sigmoid(x)
        return x
