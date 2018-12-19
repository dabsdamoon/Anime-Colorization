##### This is a model from Emil Wallner's article 'Colorizing B&W Photos with Neural Networks'
##### The author describes the model as an 'full ersion'. 
##### For more information, visit https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/

##### importing funcions

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D, add, GlobalAveragePooling2D
from keras.layers import InputLayer, Concatenate, BatchNormalization, Dropout, Activation, Dense, Flatten, RepeatVector, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

##### Define class 'fusion' for model generation (including ResNet as classification model)

class WGAN():

	def __init__(self, input_shape, latent_dim):
		
		self.img_rows = input_shape[0]
		self.img_cols = input_shape[1]
		self.img_shape_d = (input_shape[0], input_shape[1], 3)
		self.latent_dim = latent_dim

	def wasserstein_loss(y_true, y_pred):
		"""Calculates the Wasserstein loss for a sample batch.
		The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
		has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
		GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
		the discriminator wants to make the distance between its output for real and generated samples as large as possible.
		The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
		0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
		Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
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
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
    
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
    
		model.add(UpSampling2D())
		model.add(Conv2D(32, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
    
		model.add(UpSampling2D())
		model.add(Conv2D(16, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
    
		model.add(Conv2D(2, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))

		model.summary()

		img = model(combined_input)
		img = Concatenate(axis=3)([image_L, img])

		return Model([noise, image_L], img)


	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape_d, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
#		model.add(Dropout(0.25))
    
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
#		model.add(Dropout(0.25)) 
 
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
#		model.add(Dropout(0.25))
    
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
#		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		img = Input(shape=self.img_shape_d)
		validity = model(img)

		return Model(img, validity)

