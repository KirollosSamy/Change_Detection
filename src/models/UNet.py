import torch
import torch.nn
from collections import deque


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, in_channels = 1, out_channels = 2):
        super(UNet, self).__init__()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Contracting path) 
        self.downs = torch.nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])
        
        # Bottle neck (Transfer of context)
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Expansive path)
        self.ups = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),     
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            DoubleConv(128, 64),
        ])
        
        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1, padding=0) 

    def forward(self, x):
        skip_connections = deque()
        
        # Encoder
        for conv in self.downs:
            x = conv(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
            
        # Decoder    
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            concat = torch.cat((skip_connections.pop(), x), dim=1)
            x = self.ups[i+1](concat)
        
        return self.final_conv(x)