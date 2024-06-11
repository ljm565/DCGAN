import torch.nn as nn
from config import Config
import math



class Generator(nn.Module):
    def __init__(self, config:Config, color_channel:int):
        super(Generator, self).__init__()
        self.noise_init_size = config.noise_init_size
        self.g_dim = config.g_dim
        self.color_channel = color_channel
        self.middle_layer_num = int(math.log2(config.img_size)) - 3
        self.generator = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.noise_init_size, out_channels=self.g_dim*2**self.middle_layer_num, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.g_dim*2**self.middle_layer_num),
                nn.ReLU())] + 
            [nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.g_dim*2**i, out_channels=self.g_dim*2**(i-1), kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.g_dim*2**(i-1)),
                nn.ReLU()
                ) for i in range(self.middle_layer_num, 0, -1)] + 
            [nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.g_dim, out_channels=self.color_channel, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
            )]
        )

    def forward(self, x):
        for layer in self.generator:
            x = layer(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, config:Config, color_channel:int):
        super(Discriminator, self).__init__()
        self.d_dim = config.d_dim
        self.color_channel = color_channel
        self.middle_layer_num = int(math.log2(config.img_size)) - 3
        self.discriminator = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels=self.color_channel, out_channels=self.d_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2))] + 
            [nn.Sequential(
                nn.Conv2d(in_channels=self.d_dim*2**i, out_channels=self.d_dim*2**(i+1), kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.d_dim*2**(i+1)),
                nn.LeakyReLU(0.2)
                ) for i in range(self.middle_layer_num)] + 
            [nn.Sequential(
                nn.Conv2d(in_channels=self.d_dim*2**self.middle_layer_num, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )]
        )

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)
        return x.view(-1, 1).squeeze(1)