import torch



from models.basictorch import Generator, Discriminator

from constants import BATCH_SIZE, IMAGE_SIZE, LATENT_DIM

LATENT_DIM = 400

generator = Generator()
generator.load_state_dict(torch.load("saved_models\\basic\\generator_280.pth"))
generator.eval()

noise = torch.randn(25, 400)
images = generator(noise)

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(5, 5)

for i in range(5):
    for j in range(5):
        axs[i, j].imshow(np.transpose(images[i*5+j].detach().numpy(), (1, 2, 0)))
        axs[i, j].axis('off')
plt.show()

