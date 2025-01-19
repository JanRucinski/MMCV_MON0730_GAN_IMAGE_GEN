from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

import os

from constants import *

import torchvision.transforms as transforms

from dataset import get_dataset

from models.trans import *



def save_checkpoint(states,is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:",device)



generator= Generator(depth1=5, depth2=4, depth3=2, initial_size=IMAGE_SIZE//4, dim=BASE_DIM*16, heads=4, mlp_ratio=4, drop_rate=0.5)#,device = device)
generator.to(device)



discriminator = Discriminator( image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, input_channel=CHANNELS, num_classes=1,
                 dim=BASE_DIM*16, depth=7, heads=4, mlp_ratio=4,
                 drop_rate=0.)
discriminator.to(device)

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def inits_weight(m):
        if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight.data, 1.)

generator.apply(inits_weight)
discriminator.apply(inits_weight)


optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=GEN_LEARNING_RATE, betas=(BETA1, BETA2))

optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),lr=DIS_LEARNING_RATE, betas=(BETA1, BETA2))
    

gen_scheduler = LinearLrDecay(optim_gen, GEN_LEARNING_RATE, 0.0, 0, MAX_ITER * CRITIC_ITERATIONS)
dis_scheduler = LinearLrDecay(optim_dis, DIS_LEARNING_RATE, 0.0, 0, MAX_ITER * CRITIC_ITERATIONS)




writer=SummaryWriter()
writer_dict = {'writer':writer}
writer_dict["train_global_steps"]=0
writer_dict["valid_global_steps"]=0

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train(noise,generator, discriminator, optim_gen, optim_dis,
        epoch, writer, schedulers, img_size=IMAGE_SIZE, latent_dim = LATENT_DIM,
        n_critic = CRITIC_ITERATIONS,
        gener_batch_size=BATCH_SIZE, device="cuda:0"):


    writer = writer_dict['writer']
    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()

    transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = get_dataset()

    for index, (img, _) in enumerate(train_loader):

        global_steps = writer_dict['train_global_steps']

        real_imgs = img.type(torch.cuda.FloatTensor)

        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], latent_dim)))

        optim_dis.zero_grad()
        real_valid=discriminator(real_imgs)
        fake_imgs = generator(noise).detach()

        fake_valid = discriminator(fake_imgs)

        
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), PHI)
        loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 / (PHI** 2)      
     

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

        if global_steps % n_critic == 0:

            optim_gen.zero_grad()
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            gener_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

            generated_imgs= generator(gener_noise)
            fake_valid = discriminator(generated_imgs)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
            writer.add_scalar("gener_loss", gener_loss.item(), global_steps)

            gen_step += 1

        if gen_step and index % 100 == 0:
            sample_imgs = generated_imgs[:5]
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(sample_imgs, f'output/trans/generated_img_{epoch}_{index % len(train_loader)}.jpg', nrow=5, normalize=True, scale_each=True)            
            tqdm.write("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch+1, index % len(train_loader), len(train_loader), loss_dis.item(), gener_loss.item()))







best = 1e4

def noise(n_samples, z_dim, device):
        return torch.randn(n_samples,z_dim).to(device)

generator.load_state_dict(torch.load("saved_models\\trans\\generator_18.pth"))
discriminator.load_state_dict(torch.load("saved_models\\trans\\discriminator_18.pth"))
for epoch in range(100):

    lr_schedulers = (gen_scheduler, dis_scheduler) 

    train(noise, generator, discriminator, optim_gen, optim_dis,
    epoch, writer, lr_schedulers,img_size=IMAGE_SIZE, latent_dim = LATENT_DIM,
    n_critic = CRITIC_ITERATIONS,
    gener_batch_size=BATCH_SIZE)

    checkpoint = {'epoch':epoch, 'best_fid':best}
    checkpoint['generator_state_dict'] = generator.state_dict()
    checkpoint['discriminator_state_dict'] = discriminator.state_dict()
    torch.save(generator.state_dict(), "saved_models\\trans\\generator_{epoch}.pth".format(epoch=epoch))
    torch.save(discriminator.state_dict(), "saved_models\\trans\\discriminator_{epoch}.pth".format(epoch=epoch))


   


checkpoint = {'epoch':epoch, 'best_fid':best}
checkpoint['generator_state_dict'] = generator.state_dict()
checkpoint['discriminator_state_dict'] = discriminator.state_dict()
save_checkpoint(checkpoint, output_dir="saved_models", is_best=True)
