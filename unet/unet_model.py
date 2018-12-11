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

def model(image_size, kernel_size = 8, optimizer = 'adam'):

	n_classes=2 # This is the number of output channel. Since it's AB channel for us, the number equals to 2
	n_channels=1 # This si the numper of input channel, which is just 1 (L channel)
	n_filters_start=kernel_size # This is the number of filters for convolutional layer to start
	growth_factor=2 # This is the factor number indicating how much deeper convolutional layers are going to be.
	upconv=False # Whether to use Conv2dTranspose or UpSampling2D layer for up-convolution process. 

	droprate=0.25
	n_filters = n_filters_start
	inputs = Input((image_size[0], image_size[1], n_channels))
	input2 = inputs
	# input2 = BatchNormalization()(input2)
	conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(input2)
	conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	# pool1 = Dropout(droprate)(pool1)

	n_filters *= growth_factor
	# pool1 = BatchNormalization()(pool1)
	conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	# pool2 = Dropout(droprate)(pool2)

	n_filters *= growth_factor
	# pool2 = BatchNormalization()(pool2)
	conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	# pool3 = Dropout(droprate)(pool3)

	n_filters *= growth_factor
	# pool3 = BatchNormalization()(pool3)
	conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
	conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
	pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
	# pool4_1 = Dropout(droprate)(pool4_1)

	n_filters *= growth_factor
	# pool4_1 = BatchNormalization()(pool4_1)
	conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
	conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
	pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
	# pool4_2 = Dropout(droprate)(pool4_2)

	n_filters *= growth_factor
	conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
	conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

	n_filters //= growth_factor
	if upconv: # conv4_1 layer is covolutioned layer from pool4_1 layer
	    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
	else:
	    up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
	# up6_1 = BatchNormalization()(up6_1)
	conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
	conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
	# conv6_1 = Dropout(droprate)(conv6_1)

	n_filters //= growth_factor
	if upconv:
	    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
	else:
	    up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
	# up6_2 = BatchNormalization()(up6_2)
	conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
	conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
	# conv6_2 = Dropout(droprate)(conv6_2)

	n_filters //= growth_factor
	if upconv:
	    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
	else:
	    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
	# up7 = BatchNormalization()(up7)
	conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
	# conv7 = Dropout(droprate)(conv7)

	n_filters //= growth_factor
	if upconv:
	    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
	else:
	    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
	# up8 = BatchNormalization()(up8)
	conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
	# conv8 = Dropout(droprate)(conv8)

	n_filters //= growth_factor
	if upconv:
	    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
	else:
	    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
	conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(n_classes, (1, 1), activation='tanh')(conv9)

	model = Model(inputs=inputs, outputs=conv10)

	model.summary()

	model.compile(optimizer=optimizer, loss='mse')

	return model