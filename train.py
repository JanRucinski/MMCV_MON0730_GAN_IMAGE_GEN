import torch

import torchvision
from constants import LATENT_DIM, IMAGE_SIZE, BATCH_SIZE, CHANNELS
import time
from torch.autograd import Variable
from models.wgan import Generator, Discriminator
from util import calculate_gradient_penalty
print(torch.cuda.is_available())    

gpu = torch.device('cuda:0')

cpu = torch.device('cpu')

torch.backends.cuda.matmul.allow_tf32 = True

def train(epochs, dataset, generator, discriminator, start_epoch=0):
    generator.to(gpu)
    discriminator.to(gpu)
    
    #loss = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #optimizer_G = torch.optim.Adagrad(generator.parameters(), lr=0.0001, weight_decay=0.01)
    #optimizer_D = torch.optim.Adagrad(discriminator.parameters(), lr=0.0001, weight_decay=0.01)
    real_label = 0.99  # Instead of 1
    fake_label = 0.01  # Instead of 0
    generator.train()
    discriminator.train()
    #torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    for epoch in range(start_epoch,epochs+start_epoch):
        time_start = time.time()
        for i, (imgs, _) in enumerate(dataset):

            batch_size = imgs.size(0)
            if batch_size != BATCH_SIZE:
                continue
            imgs = imgs.to(gpu)
            # Train Discriminator first
            optimizer_D.zero_grad()
            
            # Add noise to images for stability
            noise_factor = max(0.0, 0.1 * (1 - epoch/epochs))
            real_imgs = imgs.to(gpu) + noise_factor * torch.randn_like(imgs.to(gpu))
            real = torch.full((batch_size, 1), real_label, device=gpu)
            fake = torch.full((batch_size, 1), fake_label, device=gpu)
            
            # Real images
            real_pred = discriminator(real_imgs)
            #real_loss = loss(real_pred, real)
            real_loss = torch.mean(real_pred)
            
            
            # Fake images
            z = torch.randn(batch_size, LATENT_DIM,1,1, device=gpu)
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs.detach())
            #fake_loss = loss(fake_pred, fake)
            fake_loss = torch.mean(fake_pred)
            
            gradient_penalty = calculate_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, gpu)
            
            
            
            
            # Train discriminator
            d_loss = fake_loss - real_loss + 10 * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Generate new fake images for generator training
            z = torch.randn(batch_size, LATENT_DIM,1,1,device=gpu)
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs)
            
            # Generator tries to fool discriminator
            g_loss = -torch.mean(fake_pred)
            g_loss.backward()
            optimizer_G.step()
            
            print(
                f"[Epoch {epoch}/{epochs+start_epoch}] [Batch {i}/{len(dataset)}] [D_fake loss: {fake_loss.item()}] [D_real loss: {real_loss.item()}] [G loss: {g_loss.item()}]"
            )
            
        
        torch.save(generator.state_dict(), f"saved_models\\wgan\\generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"saved_models\\wgan\\discriminator_{epoch}.pth")
        print(f"Time taken: {time.time() - time_start}")
        sample_images(generator, epoch, LATENT_DIM)

        
def sample_images(generator, epoch, latent_dim):
    with torch.cuda.device(0):
        z = torch.randn(5, latent_dim, 1, 1, device=gpu)
        gen_imgs = generator(z)
        torchvision.utils.save_image(gen_imgs[:5], f"output/wgan/{epoch}.png", nrow=5, normalize=True)
    
    
if __name__ == "__main__":
    with torch.cuda.device(0):
        from dataset import get_dataset
        dataset = get_dataset()
        generator = Generator()
        generator.load_state_dict(torch.load("saved_models\\wgan\\generator_20.pth"))
        discriminator = Discriminator()
        discriminator.load_state_dict(torch.load("saved_models\\wgan\\discriminator_20.pth"))
        train(300, dataset, generator, discriminator, start_epoch=20)
        torch.save(generator.state_dict(), "saved_models\\wgan\\generator.pth")
        torch.save(discriminator.state_dict(), "saved_models\\wgan\\discriminator.pth")
   