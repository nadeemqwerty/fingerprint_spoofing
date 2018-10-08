
import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import optimizers
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import os.path
import numpy as np
from PIL import Image
from numpy import *
import random
import tensorflow as tf
from keras.utils import plot_model
#import keras


def Feature_Extractor():
	input_img = Input(shape=(224,224,3))
	# conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	# conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 112 x 112

	# conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	# conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv2) # 56 x 56

	# conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	# conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 28 x 28

	# conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	# conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 14 x 14

	# conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	# conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)

	up6 = UpSampling2D((2,2))(conv5) # 28 x 28
	# up6 = merge([up6, conv4], mode='concat', concat_axis=3)
	# conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
	conv6 = BatchNormalization()(conv6)
	# conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)

	up7 = UpSampling2D((2,2))(conv6) # 56 x 56
	# up7 = merge([up7, conv3], mode='concat', concat_axis=3)
	# conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
	conv7 = BatchNormalization()(conv7)
	# conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)

	up8 = UpSampling2D((2,2))(conv7) # 112 x 112
	# up8 = merge([up8, conv2], mode='concat', concat_axis=3)
	# conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
	conv8 = BatchNormalization()(conv8)
	# conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)

	up9 = UpSampling2D((2,2))(conv8) # 224 x 224
	# up9 = merge([up9, conv1], mode='concat', concat_axis=3)
	# conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
	conv9 = BatchNormalization()(conv9)
	# conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)

	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)
	rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(loss='mean_absolute_error', optimizer=rms)
	#autoencoder.load_weights('autoencoder_2.h5')


	for i in range(21):
		autoencoder.layers.pop()
	for layers in autoencoder.layers:
		layers.trainable = False
	autoencoder.summary()
	return autoencoder


def prepare_batch(train_data):
	anchors = []
	examples = []

	for class_ in train_data:
		random_pair = random.sample(class_,2)
		anchors.append(random_pair[0])
		examples.append(random_pair[1])
	# batch = anchors + examples
	# return np.asarray(batch)
	return (np.asarray(anchors),np.asarray(examples))


def loss_function(y_true,y_pred):

	margin = 10

	y_t_norm = backend.l2_normalize(y_true, axis=1)
	y_p_norm = backend.l2_normalize(y_pred, axis=1)

	neg_matrix = backend.dot(y_t_norm,backend.transpose(y_p_norm))
	
	pos_val = backend.batch_dot(y_t_norm,y_p_norm,axes=1)
	pos_matrix = tf.tile(pos_val,[1,tf.size(pos_val)])
	pos_matrix = tf.reshape(pos_matrix,[tf.size(pos_val),tf.size(pos_val)])
	return backend.mean(backend.log(1 + backend.sum(backend.exp(neg_matrix - pos_matrix + margin),axis=1)),axis=0)


# print ("Loading 2013")

# dirs = ['live','BodyDouble','Ecoflex','Gelatin','Latex','Playdoh','WoodGlue','Modasil']
# #train_data is the list of arrays of images contained in respective folders in dirs list
# train_data = []

# dir_prefix = "../segmented/"
# for directory in dirs:
# 	temp = []
# 	for file_name in os.listdir(dir_prefix + directory):
# 		if file_name[-3:] == '.db': continue  #ignoring the Thumps.db file created automatically by ubuntu
# 		img = image.load_img(dir_prefix + directory+"/"+file_name, target_size=(224,224,3))
# 		img_array = image.img_to_array(img)
# 		temp.append(img_array)
# 	temp = np.array(temp)
# 	temp.astype('float')/255.
# 	train_data.append(temp)

# print ("LivDet2013 Loaded")


encoder = Feature_Extractor()
input_layer = encoder.input
encoder_output = encoder.layers[-1].output
x = Flatten(name='flatten')(encoder_output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(128, activation='sigmoid')(x)

model = Model(input=input_layer,output=x)
print(model.summary())



adam = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999)
sgd = optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False)
model.compile(optimizer=adam,loss=loss_function)

plot_model(model, to_file='model.png')

iterations = 10000
losses = []
min_losses=[]

# for i in range(1,iterations+1):
# 	print("iteration:",i)
# 	(anchors,examples) = prepare_batch(train_data)
# 	y_true = model.predict(examples)
# 	loss = model.train_on_batch(anchors,y_true)
# 	print(loss)	
# 	losses.append(loss)
# 	if i%50 == 0:
# 		min_losses.append(min(losses[-50:]))
# 	print(min_losses)


# print("minimum:",min(losses))
# print("average:",sum(losses)/len(losses))

# # X_train = prepare_batch(train_data)
# # for i in range(1,200):
# # 	X_train = np.concatenate((X_train,prepare_batch(train_data)),axis=0)
# # print(X_train.shape)

# # print("last layer's weights:")
# # print(model.layers[-1].get_weights())

# # model.fit(np.asarray(X_train), np.ones((16*200,300)), batch_size=16, nb_epoch=50,
# #           verbose=1, validation_split=0.2)


# model.save_weights('n_pair.h5')


#print(losses)
#check model accuracy.	


