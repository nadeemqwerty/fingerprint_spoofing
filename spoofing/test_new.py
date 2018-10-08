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
from random import *
import keras
from sklearn.metrics import pairwise

def loss_function(y_true,y_pred):

	margin = 2

	y_t_norm = backend.l2_normalize(y_true, axis=1)
	y_p_norm = backend.l2_normalize(y_pred, axis=1)

	neg_matrix = backend.dot(y_t_norm,backend.transpose(y_p_norm))
	
	pos_val = backend.batch_dot(y_t_norm,y_p_norm,axes=1)
	pos_matrix = tf.tile(pos_val,[1,tf.size(pos_val)])
	pos_matrix = tf.reshape(pos_matrix,[tf.size(pos_val),tf.size(pos_val)])
	return backend.mean(backend.log(backend.exp(backend.sum(tf.maximum(neg_matrix + margin - pos_matrix,0),axis=1))),axis=0)


print ("Loading 2013")

dirs = ['live','BodyDouble','Ecoflex','Gelatin','Latex','Playdoh','WoodGlue','Modasil']
#train_data is the list of arrays of images contained in respective folders in dirs list
train_data = []

dir_prefix = "./segmented/"
for directory in dirs:
	temp = []
	for file_name in os.listdir(dir_prefix + directory):
		if file_name[-3:] == '.db': continue  #ignoring the Thumps.db file created automatically by ubuntu
		img = image.load_img(dir_prefix + directory+"/"+file_name, target_size=(224,224,3))
		img_array = image.img_to_array(img)
		temp.append(img_array)
	temp = np.array(temp)
	temp.astype('float')/255.
	train_data.append(temp)

print ("LivDet2013 Loaded")

model = VGG16(include_top=True, weights='imagenet', classes=1000)
model.layers.pop()
model.layers[-1].outbound_nodes = []
inp = model.input
opt = model.layers[-1].output
opt2 = Dense(250, activation = 'sigmoid')(opt)     #intialization?
model = Model(input = inp, output = opt2)

adam = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999)
sgd = optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False)
model.compile(optimizer=adam,loss=loss_function)

model.load_weights('n_pair.h5')


for i in range(1,11):
	random_number = randint(1,3000)
	live_image = train_data[0][random_number]
	#print("live images shape:",live_image.shape)
	random_images = []
	for i in range(0,8):
		random_images.append(train_data[i][randint(1,200)])

	images = np.asarray(random_images)
	#print("images shape:",images.shape)

	images = np.insert(images,8,live_image,axis=0)
	#print("images shape:",images.shape)

	embeddings	= model.predict(images)
	#print(embeddings.shape)
	#print(embeddings)
	for i in range(0,8):
		print np.linalg.norm(embeddings[8]-embeddings[i]),
	print
	







