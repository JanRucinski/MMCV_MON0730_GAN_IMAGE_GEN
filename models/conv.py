import torch

from torch.nn import Module, Linear, LeakyReLU, Sequential, Tanh, Sigmoid, Unflatten, Conv2d, ConvTranspose2d, BatchNorm2d, Flatten

from constants import LATENT_DIM, IMAGE_SIZE, BATCH_SIZE, CHANNELS


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = Sequential(
            # Input is the latent vector Z
            ConvTranspose2d(LATENT_DIM, IMAGE_SIZE * 8, 4, 1, 1, bias=False),
            BatchNorm2d(IMAGE_SIZE * 8),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(IMAGE_SIZE * 8, IMAGE_SIZE * 4, 4, 2, 1, bias=False),
            BatchNorm2d(IMAGE_SIZE * 4),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(IMAGE_SIZE * 4, IMAGE_SIZE * 2, 4, 2, 1, bias=False),
            BatchNorm2d(IMAGE_SIZE * 2),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(IMAGE_SIZE * 2, IMAGE_SIZE, 4, 2, 1, bias=False),
            BatchNorm2d(IMAGE_SIZE),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(IMAGE_SIZE, CHANNELS, 4, 2, 1, bias=False),
            Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
    

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = Sequential(
            # Input is the image
            Conv2d(CHANNELS, IMAGE_SIZE,2, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv2d(IMAGE_SIZE, IMAGE_SIZE * 2, 2, 2, 1, bias=False),
            BatchNorm2d(IMAGE_SIZE * 2),
            LeakyReLU(0.2, inplace=True),
            Conv2d(IMAGE_SIZE * 2, IMAGE_SIZE * 4, 2, 2, 1, bias=False),
            BatchNorm2d(IMAGE_SIZE * 4),
            LeakyReLU(0.2, inplace=True),
            Conv2d(IMAGE_SIZE * 4, IMAGE_SIZE * 8, 4, 2, 1, bias=False),
            BatchNorm2d(IMAGE_SIZE * 8),
            LeakyReLU(0.2, inplace=True),
            Conv2d(IMAGE_SIZE * 8, 1, 4, 1, 0, bias=False),
            Flatten(),
            Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    