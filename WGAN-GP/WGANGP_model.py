##### This is a model from Emil Wallner's article 'Colorizing B&W Photos with Neural Networks'
##### The author describes the model as an 'full ersion'. 
##### For more information, visit https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/

##### importing funcions

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D, add, GlobalAveragePooling2D
from keras.layers import InputLayer, Concatenate, BatchNormalization, Dropout, Activation, Dense, Flatten, RepeatVector, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import _Merge
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

from functools import partial
import numpy as np

##### Define class 'fusion' for model generation (including ResNet as classification model)

batch_size = 32

##### Define class 'fusion' for model generation (including ResNet as classification model)

class RandomWeightedAverage(_Merge):
    
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        epsilon = K.random_uniform((batch_size, 1, 1, 1))
        return (epsilon * inputs[0]) + ((1 - epsilon) * inputs[1])

    
class WGANGP():

    def __init__(self, input_shape, latent_dim):

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_shape_d = (input_shape[0], input_shape[1], 3)
        self.img_shape_l = (input_shape[0], input_shape[1], 1)
        self.latent_dim = latent_dim
        
        # Build generator and discriminator
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        ##### computational graph for discriminator
        
        self.generator.trainable = False
        
        # defining inputs
        real_img = Input(shape = self.img_shape_d)
        img_l = Input(shape = (self.img_shape_l))
        z_disc = Input(shape = (self.latent_dim,))
        
        fake_img_d = self.generator([z_disc, img_l])
        
        fake = self.discriminator(fake_img_d) # discriminator output layer of fake images
        valid = self.discriminator(real_img) # discriminator output layer of real images

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()(inputs = [real_img, fake_img_d]) # calculation interpolation xhat 
        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_img) # D(xhat)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        optimizer_d = Adam(0.0002, 0.5)
#         optimizer_d = RMSprop(lr=0.00005)
    
        self.discriminator_model = Model(inputs=[real_img, z_disc, img_l],
                            outputs=[valid, fake, validity_interpolated])
        
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer_d,
                                        loss_weights=[1, 1, 10])
        
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the discriminator's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        
        # Generate images based of noise
        fake_img_g = self.generator([z_gen, img_l])
        
        # Discriminator determines validity
        valid = self.discriminator(fake_img_g)
        
        # Defines generator model
        optimizer_g = Adam(0.0002, 0.5)
#         optimizer_g = RMSprop(lr=0.00005)
        self.generator_model = Model([z_gen, img_l], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer_g)
        
        
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)    

    def wasserstein_loss(self,y_true, y_pred):

        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        image_L = Input(shape=(self.img_rows,self.img_cols,1))

        noise_h = Dense(128 * 8 * 8, activation="relu")(noise)
        noise_h = Reshape((8, 8, 128))(noise_h)

        image_L_h = Reshape((8, 8, 256))(image_L)

        combined_input = Concatenate(axis=3)([noise_h, image_L_h])

        model = Sequential()

        model.add(UpSampling2D(input_shape = (8,8,128+256)))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(2, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        img = model(combined_input)
        img = Concatenate(axis=3)([image_L, img])

        return Model([noise, image_L], img)


    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(8, kernel_size=3, strides=2, input_shape=self.img_shape_d, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25)) 

        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1))

        print('------------------------------')
        print('Discriminator structure:')
        print('------------------------------')
        model.summary()

        img = Input(shape=self.img_shape_d)
        validity = model(img)

        return Model(img, validity)
