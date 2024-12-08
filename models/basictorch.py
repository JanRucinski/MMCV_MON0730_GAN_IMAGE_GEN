import torch
from torch.nn import Module, Linear, LeakyReLU, Sequential, Tanh, Sigmoid, Unflatten
from constants import LATENT_DIM, IMAGE_SIZE, BATCH_SIZE, CHANNELS

class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = Sequential(
            Linear(LATENT_DIM, LATENT_DIM*5),
            LeakyReLU(),
            Linear(LATENT_DIM*5, LATENT_DIM*10),
            LeakyReLU(),
            Linear(LATENT_DIM*10 , LATENT_DIM*20),
            LeakyReLU(),
            Linear(LATENT_DIM*20, IMAGE_SIZE * IMAGE_SIZE * CHANNELS),
            Tanh(),
            Unflatten(1, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        )

    def forward(self, x):
        return self.model(x)
    

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.model = Sequential(
            Linear(IMAGE_SIZE * IMAGE_SIZE * CHANNELS, LATENT_DIM*10),
            LeakyReLU(),
            Linear(LATENT_DIM*10, LATENT_DIM*5),
            LeakyReLU(),
            Linear(LATENT_DIM*5, 1),
            Sigmoid()
        )

    def forward(self, x):
        return self.model(self.flatten(x))