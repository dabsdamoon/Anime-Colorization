##### This is a model from Emil Wallner's article 'Colorizing B&W Photos with Neural Networks'
##### The author describes the model as an 'full ersion'. 
##### For more information, visit https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/

##### importing funcions

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D, add, GlobalAveragePooling2D
from keras.layers import InputLayer, concatenate, BatchNormalization, Dropout, Activation, Dense, Flatten, RepeatVector, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

##### Define class 'fusion' for model generation (including ResNet as classification model)

class fusion():

	def __init__(self, input_shape, channel_last, classes):
		
		self.input_shape = input_shape
		self.channel_last = channel_last
		self.classes = classes


	def identity_block(self, input_tensor, kernel_size, filters, stage, block):
		"""The identity block is the block that has no conv layer at shortcut.
		# Arguments
			input_tensor: input tensor
			kernel_size: default 3, the kernel size of
				middle conv layer at main path
			filters: list of integers, the filters of 3 conv layer at main path
			stage: integer, current stage label, used for generating layer names
			block: 'a','b'..., current block label, used for generating layer names
		# Returns
			Output tensor for the block.
		"""
		filters1, filters2, filters3 = filters # this indicates the number of filter (here, assume that all filter number are same)

		bn_axis = self.channel_last # number of classes being classified
    
		# names for the layers
		conv_name_base = 'res' + str(stage) + block + '_branch'
		bn_name_base = 'bn' + str(stage) + block + '_branch'

		# model definition
		x = Conv2D(filters1, (1, 1),
						  kernel_initializer='he_normal',
						  name=conv_name_base + '2a')(input_tensor)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters2, kernel_size,
						  padding='same',
						  kernel_initializer='he_normal',
						  name=conv_name_base + '2b')(x)
    
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters3, (1, 1),
						  kernel_initializer='he_normal',
						  name=conv_name_base + '2c')(x)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

		# Here, we add the outcome tensor(layer) x to original input tensor
		x = add([x, input_tensor])
		x = Activation('relu')(x)
    
		return x


	def conv_block(self,
				   input_tensor,
				   kernel_size,
				   filters,
				   stage,
				   block,
				   strides=(2, 2)):
		"""A block that has a conv layer at shortcut.
		# Arguments
			input_tensor: input tensor
			kernel_size: default 3, the kernel size of
				middle conv layer at main path
			filters: list of integers, the filters of 3 conv layer at main path
			stage: integer, current stage label, used for generating layer names
			block: 'a','b'..., current block label, used for generating layer names
			strides: Strides for the first conv layer in the block.
		# Returns
			Output tensor for the block.
		Note that from stage 3,
		the first conv layer at main path is with strides=(2, 2)
		And the shortcut should have strides=(2, 2) as well
		"""
		# number of filters
		filters1, filters2, filters3 = filters

		# number of classification categories
		bn_axis = self.channel_last
    
		# names for the layers
		conv_name_base = 'res' + str(stage) + block + '_branch'
		bn_name_base = 'bn' + str(stage) + block + '_branch'

		# simple convolutional layers
		x = Conv2D(filters1, (1, 1), strides=strides,
						  kernel_initializer='he_normal',
						  name=conv_name_base + '2a')(input_tensor)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters2, kernel_size, padding='same',
						  kernel_initializer='he_normal',
						  name=conv_name_base + '2b')(x)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters3, (1, 1),
						  kernel_initializer='he_normal',
						  name=conv_name_base + '2c')(x)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

		# define a shortcut, which is an output for convolutional layer obtained by directly applying theninput tensor to filter 3
		shortcut = Conv2D(filters3, (1, 1), strides=strides,
								 kernel_initializer='he_normal',
								 name=conv_name_base + '1')(input_tensor)
    
		shortcut = BatchNormalization(
			axis=bn_axis, name=bn_name_base + '1')(shortcut)

		# adding stepwise output x to shortcut output
		x = add([x, shortcut])
		x = Activation('relu')(x)
    
		return x


	def resnet(self):

		Inputs = Input(self.input_shape)

		resnet = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(Inputs)
		resnet = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(resnet)
		resnet = BatchNormalization(axis= self.channel_last, name='bn_conv1')(resnet)
		resnet = Activation('relu')(resnet)
		resnet = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(resnet)
		resnet = MaxPooling2D((3, 3), strides=(2, 2))(resnet)

		resnet = self.conv_block(resnet, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
		resnet = self.identity_block(resnet, 3, [64, 64, 256], stage=2, block='c')
		resnet = self.identity_block(resnet, 3, [64, 64, 256], stage=2, block='b')

		resnet = self.conv_block(resnet, 3, [128, 128, 512], stage=3, block='a')
		resnet = self.identity_block(resnet, 3, [128, 128, 512], stage=3, block='b')
		resnet = self.identity_block(resnet, 3, [128, 128, 512], stage=3, block='c')
		resnet = self.identity_block(resnet, 3, [128, 128, 512], stage=3, block='d')

		resnet = self.conv_block(resnet, 3, [256, 256, 1024], stage=4, block='a')
		resnet = self.identity_block(resnet, 3, [256, 256, 1024], stage=4, block='b')
		resnet = self.identity_block(resnet, 3, [256, 256, 1024], stage=4, block='c')
		resnet = self.identity_block(resnet, 3, [256, 256, 1024], stage=4, block='d')
		resnet = self.identity_block(resnet, 3, [256, 256, 1024], stage=4, block='e')
		resnet = self.identity_block(resnet, 3, [256, 256, 1024], stage=4, block='f')

		resnet = self.conv_block(resnet, 3, [512, 512, 2048], stage=5, block='a')
		resnet = self.identity_block(resnet, 3, [512, 512, 2048], stage=5, block='b')
		resnet = self.identity_block(resnet, 3, [512, 512, 2048], stage=5, block='c')

		resnet = GlobalAveragePooling2D(name='avg_pool')(resnet)
		resnet = Dense(self.classes, activation='softmax', name='fc1000')(resnet)

		resnet = Model(Inputs, resnet)

		print(resnet.summary())

		return resnet


	# The structure of colorization model is from 'Alpha' in Emil Wallner's Article

	def model(self, kernel_size = 4, optimizer = Adam(0.0002, 0.5)):

		embed_input = Input(shape=(self.classes,))

		#Encoder
		encoder_input = Input(shape=(self.input_shape[0], self.input_shape[1], 1,))
		encoder_output = Conv2D(kernel_size, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
		encoder_output = Conv2D(kernel_size*2, (3,3), activation='relu', padding='same')(encoder_output)
		encoder_output = Conv2D(kernel_size*4, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
		encoder_output = Conv2D(kernel_size*4, (3,3), activation='relu', padding='same')(encoder_output)
		encoder_output = Conv2D(kernel_size*8, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
		encoder_output = Conv2D(kernel_size*8, (3,3), activation='relu', padding='same')(encoder_output)
		# encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

		#Fusion
		fusion_output = RepeatVector(16 * 16)(embed_input) 
		fusion_output = Reshape(([16, 16, self.classes]))(fusion_output)
		fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
		fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

		#Decoder
		decoder_output = Conv2D(kernel_size*16, (3,3), activation='relu', padding='same')(fusion_output)
		decoder_output = UpSampling2D((2, 2))(decoder_output)
		decoder_output = Conv2D(kernel_size*16, (3,3), activation='relu', padding='same')(decoder_output)
		decoder_output = UpSampling2D((2, 2))(decoder_output)

		decoder_output = Conv2D(kernel_size*8, (3,3), activation='relu', padding='same')(decoder_output)
		decoder_output = Conv2D(kernel_size*4, (3,3), activation='relu', padding='same')(decoder_output)
		decoder_output = Conv2D(kernel_size*2, (3,3), activation='relu', padding='same')(decoder_output)
		decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
		decoder_output = UpSampling2D((2, 2))(decoder_output)

		model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

		model.compile(optimizer= optimizer, loss='mse')

		model.summary()

		return model

