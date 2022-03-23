import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

class tanh_discr(torch.autograd.Function):
    '''
    Implementation of discretized 𝑡𝑎𝑛ℎ activation functions.
    Returns value from -1 to +1
    In backward pass behaves as ordinary tanh

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - n - discretization parameter

    References:
        - See related paper:
        https://arxiv.org/pdf/2105.01768.pdf

    Examples:
        >>> x = torch.randn(256)
        >>> x = tanh_discr.apply(x, 256)
    '''
    @staticmethod
    def forward(ctx, x, n=2):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        th = torch.tanh(x)
        ctx.save_for_backward(th)
        _n = (n-1)/2
        return torch.round(_n + _n * th) / _n - 1
    
    @staticmethod
    def backward(ctx, grad_output):
        th, = ctx.saved_tensors
        return (1 - th ** 2) * grad_output, None

class PreEncoder(nn.Module):
    def __init__(self, num_layers=6, num_channels=64, kernel_size=5):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, num_channels, kernel_size, padding='same')])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_channels, num_channels, kernel_size, padding='same'))

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x))
        return x

class DownDiscretizationEncoder(nn.Module):
    def __init__(self, kernel_size=5, input_channels = 64):
        super().__init__()
        num_layers=8
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, 1, kernel_size, padding='same')])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(1, 1, kernel_size, padding='same'))
        
    def forward(self, x):
        for removed_bits, conv in enumerate(self.convs):
            x = tanh_discr.apply(conv(x), 2 ** (8 - removed_bits))
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers=2, num_channels=64, kernel_size=5):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, num_channels, kernel_size, padding='same')])
        for i in range(num_layers - 2):
            self.convs.append(nn.Conv2d(num_channels, num_channels, kernel_size, padding='same'))
        self.convs.append(nn.Conv2d(num_channels, 3, kernel_size, padding='same'))

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x))
        x = torch.tanh(self.convs[-1](x))
        return x

class TexturesModel(nn.Module):
    def __init__(self, num_layers=6, num_channels=64, kernel_size=5, alpha=0.1, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pre_encoder = PreEncoder(num_layers, num_channels, kernel_size=kernel_size)
        self.dde = DownDiscretizationEncoder(input_channels=num_channels, kernel_size=kernel_size)
        self.decoder = Decoder(num_layers, num_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.pre_encoder(x)
        x = self.dde(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('./model/model.torch', map_location=torch.device(device))
input_dir = Path('./input_img')
totesor = transforms.ToTensor()
topil = transforms.ToPILImage()
total = len(list(input_dir.iterdir()))
for filename in tqdm(input_dir.iterdir(), total=total):
    image = Image.open(filename)
    encoded = model(totesor(image).to(device) * 2 - 1) / 2 + 0.5
    encoded = topil(encoded.cpu().detach()[0])
    if encoded.mode != 'RGB':
        encoded = encoded.convert('RGB')
    encoded.save("./output_img/encoded" + str(filename.name))