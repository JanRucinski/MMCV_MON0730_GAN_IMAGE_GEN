import torch

import torchvision
from constants import LATENT_DIM, IMAGE_SIZE, BATCH_SIZE, CHANNELS
import time
from torch.autograd import Variable
print(torch.cuda.is_available())    

gpu = torch.device('cuda:0')



def train(epochs, dataset, generator, discriminator):
    generator.to(gpu)
    discriminator.to(gpu)
    with torch.device(gpu):
        loss = torch.nn.BCELoss()
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        real_label = 0.99  # Instead of 1
        fake_label = 0.01  # Instead of 0

        for epoch in range(epochs):
            time_start = time.time()
            for i, (imgs, _) in enumerate(dataset):
                batch_size = imgs.size(0)
            
                # Train Discriminator first
                optimizer_D.zero_grad()
                
                # Add noise to images for stability
                noise_factor = max(0.0, 0.1 * (1 - epoch/epochs))
                real_imgs = imgs.to(gpu) + noise_factor * torch.randn_like(imgs.to(gpu))
                real = torch.full((batch_size, 1), real_label, device=gpu)
                fake = torch.full((batch_size, 1), fake_label, device=gpu)
                
                # Real images
                real_pred = discriminator(real_imgs)
                real_loss = loss(real_pred, real)
                
                # Fake images
                z = torch.randn(batch_size, LATENT_DIM, device=gpu)
                fake_imgs = generator(z)
                fake_pred = discriminator(fake_imgs.detach())
                fake_loss = loss(fake_pred, fake)
                
                # Train discriminator
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                
                # Generate new fake images for generator training
                z = torch.randn(batch_size, LATENT_DIM, device=gpu)
                fake_imgs = generator(z)
                fake_pred = discriminator(fake_imgs)
                
                # Generator tries to fool discriminator
                g_loss = loss(fake_pred, real)
                g_loss.backward()
                optimizer_G.step()
                
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataset)}] [D_fake loss: {fake_loss.item()}] [D_real loss: {real_loss.item()}] [G loss: {g_loss.item()}]"
                )
            print(f"Time taken: {time.time() - time_start}")
            sample_images(generator, epoch, LATENT_DIM)

        
def sample_images(generator, epoch, latent_dim):
    with torch.cuda.device(0):
        z = torch.randn(5, latent_dim)
        gen_imgs = generator(z)
        torchvision.utils.save_image(gen_imgs[:5], f"output/{epoch}.png", nrow=5, normalize=True)
    
    
if __name__ == "__main__":
    with torch.cuda.device(0):
        from models.basictorch import Generator, Discriminator
        from dataset import get_dataset
        dataset = get_dataset()
        generator = Generator()
        discriminator = Discriminator()
        train(40, dataset, generator, discriminator)
        torch.save(generator.state_dict(), "generator.pth")
        torch.save(discriminator.state_dict(), "discriminator.pth")
   