##### This is a model from Emil Wallner's article 'Colorizing B&W Photos with Neural Networks'
##### The author describes the model as an 'Alpha Version'. 
##### For more information, visit https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/

##### importing funcions
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D
from keras.layers import InputLayer, concatenate, BatchNormalization, Dropout, Activation, Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

##### model 
def model(input_shape, kernel_size, optimizer = 'adam'):

	model = Sequential()

	model.add(Conv2D(kernel_size, (3,3), input_shape = input_shape,  activation='relu', padding='same', strides = 2))
	model.add(Conv2D(kernel_size * 2, (3,3), input_shape = input_shape,  activation='relu', padding='same'))

	model.add(Conv2D(kernel_size * 4, (3,3), input_shape = input_shape,  activation='relu', padding='same', strides = 2))
	model.add(Conv2D(kernel_size * 4, (3,3), input_shape = input_shape,  activation='relu', padding='same'))

	model.add(Conv2D(kernel_size * 8, (3,3), input_shape = input_shape,  activation='relu', padding='same', strides = 2))
	model.add(Conv2D(kernel_size * 8, (3,3), input_shape = input_shape,  activation='relu', padding='same'))

	model.add(Conv2D(kernel_size * 16, (3,3), input_shape = input_shape,  activation='relu', padding='same'))

	model.add(UpSampling2D((2,2)))
	model.add(Conv2D(kernel_size * 16, (3,3), input_shape = input_shape,  activation='relu', padding='same'))

	model.add(UpSampling2D((2,2)))

	model.add(Conv2D(kernel_size * 8, (3,3), input_shape = input_shape,  activation='relu', padding='same'))
	model.add(Conv2D(kernel_size * 4, (3,3), input_shape = input_shape,  activation='relu', padding='same'))
	model.add(Conv2D(kernel_size * 2, (3,3), input_shape = input_shape,  activation='relu', padding='same'))

	model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))
	model.add(UpSampling2D((2,2)))

	model.compile(optimizer= optimizer, loss='mse')

	print(model.summary())

	return model
