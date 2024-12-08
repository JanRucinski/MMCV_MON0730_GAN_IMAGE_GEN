import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets
from constants import BATCH_SIZE, IMAGE_SIZE, LATENT_DIM

img_path = "images\\input\\data"


def get_dataset():
    with torch.device('cuda:0'):
        dataset = datasets.ImageFolder(
            root=img_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(IMAGE_SIZE),
                torchvision.transforms.CenterCrop(IMAGE_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator('cuda:0').manual_seed(2137))
        return dataset