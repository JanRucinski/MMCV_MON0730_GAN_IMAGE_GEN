import torch
from models.wgan import Generator
import torchvision
import random

def generate():
    # Load the model
    generator = Generator()
    state_dict = torch.load("saved_models\\wgan\\generator_70.pth")
    generator.load_state_dict(state_dict)
    generator.eval()  # Set the generator to evaluation mode
    
    # Generate a random noise tensor
    noise = torch.randn(1, 4096, 1, 1)
    
    # Generate an image from the noise tensor
    with torch.no_grad():  # Disable gradient calculation
        generated_image = generator(noise)
    
    return generated_image

def generate_and_save():
    generated_image = generate()
    
    # Save the generated image
    torchvision.utils.save_image(generated_image, f"img\\output_{random.randint(0,20000)}.png", normalize=True)
    
    return generated_image

