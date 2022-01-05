import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.convt_2d_1 = nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0)
        self.batch_norm_1 = nn.BatchNorm2d(512)
        self.convt_2d_2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.batch_norm_2 = nn.BatchNorm2d(256)
        self.convt_2d_3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.convt_2d_4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.batch_norm_4 = nn.BatchNorm2d(64)
        self.convt_2d_5 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.batch_norm_5 = nn.BatchNorm2d(32)
        self.convt_2d_6 = nn.ConvTranspose2d(32, output_dim, 4, 2, 1)

    def forward(self, x):
        x = self.convt_2d_1(x)
        x = self.batch_norm_1(x)
        x = torch.nn.functional.relu(x)
        x = self.convt_2d_2(x)
        x = self.batch_norm_2(x)
        x = torch.nn.functional.relu(x)
        x = self.convt_2d_3(x)
        x = self.batch_norm_3(x)
        x = torch.nn.functional.relu(x)
        x = self.convt_2d_4(x)
        x = self.batch_norm_4(x)
        x = torch.nn.functional.relu(x)
        x = self.convt_2d_5(x)
        x = self.batch_norm_5(x)
        x = torch.nn.functional.relu(x)
        x = self.convt_2d_6(x)
        x = torch.tanh(x)
        return x