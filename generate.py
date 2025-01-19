import torch
from models.wgan import Generator
import torchvision
import random
def generate():
    
    # Load the model
    model = torch.load("saved_models\\wgan\\generator_30.pth")
    
    generator = Generator()
    
    model = generator.load_state_dict(model)
    # Generate a random noise tensor
    noise = torch.randn(1, 4096, 1, 1)
    
    # Generate an image from the noise tensor
    generated_image = model(noise)
    
    return generated_image


def generate_and_save():
    generated_image = generate()
    
    
    # Save the generated image
    torchvision.utils.save_image(generated_image, f"img\\output_{random.randint(0,20000)}.png", normalize=True)
    
    
    return generated_image

