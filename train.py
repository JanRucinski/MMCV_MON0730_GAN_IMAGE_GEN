import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.basic import create_discriminator, create_generator
from dataset import make_dataset

from constants import BATCH_SIZE, IMAGE_SIZE, LATENT_DIM, TOTAL_SAMPLES





def train(epochs, discriminator, generator ):
    optimizer_discriminator = keras.optimizers.Adadelta(0.9)
    optimizer_combined = keras.optimizers.Adadelta(learning_rate=0.9)
    
    valid = tf.ones(TOTAL_SAMPLES)
    fake = np.zeros(TOTAL_SAMPLES)
    dataset = make_dataset()
    dataset_iter = iter(dataset)

    # Compile the standalone discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_discriminator, metrics=['accuracy'])

    # Create a non-trainable clone of discriminator for the combined model
    discriminator_for_combined = keras.models.clone_model(discriminator)
    discriminator_for_combined.set_weights(discriminator.get_weights())
    discriminator_for_combined.trainable = False

    # Build and compile the combined model
    
    
    combined = keras.models.Sequential(
        [generator, discriminator_for_combined]
    )
    combined.compile(loss='binary_crossentropy', optimizer=optimizer_combined)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        

        try:

            noise = tf.random.normal((TOTAL_SAMPLES, LATENT_DIM))     
            dataset.shuffle(10000)
            gen_imgs = generator.predict_on_batch(noise)
            
            discriminator.trainable = True 
            d_loss_real = discriminator.fit(dataset, verbose=1, epochs=1, steps_per_epoch=100, batch_size=BATCH_SIZE)
            repeat = True
            while repeat:
                d_loss_fake = discriminator.fit(gen_imgs, fake, verbose=1,epochs=1, batch_size=BATCH_SIZE)
                repeat = d_loss_fake.history["loss"][0] > 0.5
            d_loss = 0.5 * np.add(d_loss_real.history["loss"][0], d_loss_fake.history["loss"][0])
            d_acc = 0.5 * np.add(d_loss_real.history["accuracy"][0], d_loss_fake.history["accuracy"][0])
        
            
                
            discriminator.trainable = False
            discriminator_for_combined.set_weights(discriminator.get_weights())
            
            repeat = True
            while repeat:
                noise = tf.random.normal((TOTAL_SAMPLES, LATENT_DIM))
                g_loss = combined.fit(noise, valid, verbose=1,epochs=1,  batch_size=BATCH_SIZE).history["loss"][-1]
                repeat = g_loss > 1
            print(f"{epoch} [D loss: {d_loss} acc: {100 * d_acc} [G loss: {g_loss}]")
        except Exception as e:
            print(e)
            continue      
        if epoch % 1 == 0:
            save_imgs(epoch, generator,[ x for x in iter(dataset.take(1))])
        if epoch % 10 == 0:
            generator.save(f".\\saved_models\\generator_{epoch}.keras")
            discriminator.save(f".\\saved_models\\discriminator_{epoch}.keras")
    
    print("Training complete")
            
def save_imgs(epoch, generator, img):
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
    #loaded_generator = keras.models.load_model(".\\saved_models\\generator_40.keras")
    #loaded_discriminator = keras.models.load_model(".\\saved_models\\discriminator_40.keras")
    train(epochs=1000, discriminator=create_discriminator(), generator=create_generator()) 
except Exception as e:
    print(e)
    print("An error occurred during training")
    input("Press enter to continue...")