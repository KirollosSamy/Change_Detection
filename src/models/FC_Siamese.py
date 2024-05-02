import torch
import torch.nn
from collections import deque

class BasicConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, p=0.2) -> None:
        super(BasicConv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),   
            torch.nn.Dropout2d(p)         
        )
        
    def forward(self, x):
        return self.conv(x)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        
        mid_channels = mid_channels if mid_channels is not None else out_channels
        
        self.conv = torch.nn.Sequential(
            BasicConv(in_channels, mid_channels),
            BasicConv(mid_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)
    
class TripleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(TripleConv, self).__init__()
        
        mid_channels = mid_channels if mid_channels is not None else out_channels
        
        self.conv = torch.nn.Sequential(
            BasicConv(in_channels, mid_channels),
            BasicConv(mid_channels, mid_channels),
            BasicConv(mid_channels, out_channels),   
        )

    def forward(self, x):
        return self.conv(x)

class FC_Siamese(torch.nn.Module):
    def __init__(self, in_channels = 2, out_channels = 1):
        super(FC_Siamese, self).__init__()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Contracting path) 
        self.downs = torch.nn.ModuleList([
            DoubleConv(in_channels, 16),
            DoubleConv(16, 32),
            TripleConv(32, 64),
            TripleConv(64, 128),
        ])
        
        # Decoder (Expansive path)
        self.ups = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            TripleConv(256, 64, 128),     
            torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            TripleConv(128, 32, 64),
            torch.nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            DoubleConv(64, 16, 32),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2), 
            BasicConv(32, 16)
        ])
        
        self.final_conv = torch.nn.Conv2d(16, out_channels, kernel_size=1, padding=0) 
        # self.final_conv = torch.nn.Conv2d(16, out_channels, kernel_size=3, padding=1) 

    def forward(self, A, B):
        skip_connections = deque()
        
        # Encoder
        for conv in self.downs:
            A = conv(A)
            B = conv(B)
            skip_connections.append(torch.abs(A-B))
            A = self.pool(A)
            B = self.pool(B)
            
        # Decoder    
        for i in range(0, len(self.ups), 2):
            A = self.ups[i](A)
            concat = torch.cat((skip_connections.pop(), A), dim=1)
            A = self.ups[i+1](concat)
        
        return self.final_conv(A)
    