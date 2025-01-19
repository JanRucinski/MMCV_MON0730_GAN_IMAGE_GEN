import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 256, 4, 2, 1, bias=False),  # 64 -> 256
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 128 -> 512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # 256 -> 1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),  # 512 -> 2048
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2048, 1, 4, 1, 0),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.main(input)
        out = torch.flatten(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, 4, 1, 0, bias=False),  # 1024 -> 4096
            nn.BatchNorm2d(2048),
            nn.ReLU(True),

            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),  # 512 -> 2048
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # 256 -> 1024
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 128 -> 512
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.main(input)
        return out