import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.basic import create_discriminator, create_generator
from dataset import make_dataset

from constants import BATCH_SIZE, IMAGE_SIZE, LATENT_DIM





def train(epochs, discriminator, generator ):
    optimizer = keras.optimizers.Adadelta()
    optimizer_discriminator = keras.optimizers.Adam(0.0002, 0.5)
    optimizer_combined = keras.optimizers.Adadelta()
    
    valid = tf.ones((BATCH_SIZE, 1))
    fake = tf.zeros((BATCH_SIZE, 1))
    dataset = make_dataset()
    dataset_iter = iter(dataset)

    # Compile the standalone discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_discriminator, metrics=['accuracy'])

    # Create a non-trainable clone of discriminator for the combined model
    discriminator_for_combined = keras.models.clone_model(discriminator)
    discriminator_for_combined.set_weights(discriminator.get_weights())
    discriminator_for_combined.trainable = False

    # Build and compile the combined model
    input_tensor = keras.Input(shape=(LATENT_DIM,))
    generated_imgs = generator(input_tensor)
    validity = discriminator_for_combined(generated_imgs)
    combined = keras.models.Model(input_tensor, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer_combined)
    steps_per_epoch = 25933 // BATCH_SIZE

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        for step in range(steps_per_epoch):
            try:
                imgs = next(dataset_iter)
            except StopIteration:
                # Reinitialize the iterator if it gets exhausted (shouldn't happen with repeat())
                dataset_iter = iter(dataset)
                imgs = next(dataset_iter)
            try:
                if len(imgs) != BATCH_SIZE:
                    print("Batch size mismatch")
                    continue
                
                noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))                
                gen_imgs = generator.predict(noise)
                discriminator.trainable = True 
                d_loss_real = discriminator.train_on_batch(imgs, valid)
                d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
                
                noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))    
                discriminator.trainable = False

                g_loss = combined.train_on_batch(noise, valid)
                
                print(f"{epoch} [D loss: {d_loss[0]} acc: {100 * d_loss[1]}] [G loss: {g_loss}]")
            except Exception as e:
                print(e)
                continue      
        if epoch % 1 == 0:
            save_imgs(epoch, generator)
        if epoch % 100 == 0:
            generator.save(f".\\saved_models\\generator_{epoch}.keras")
            discriminator.save(f".\\saved_models\\discriminator_{epoch}.keras")
    
    print("Training complete")
            
def save_imgs(epoch, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
    gen_imgs = generator.predict(noise)
    
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f".\\output\\{epoch}.png")
    plt.close()

try:
    train(epochs=10000, discriminator=create_discriminator(), generator=create_generator()) 
except Exception as e:
    print(e)
    print("An error occurred during training")
    input("Press enter to continue...")