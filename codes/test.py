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

def distance(anchor,positive_example,negative_example):
	# return tf.reduce_sum(tf.square(tf.subtract(anchor,positive_example))) - tf.reduce_sum(tf.square(tf.subtract(anchor,negative_example)))
	# return backend.sum(anchor*positive_example - anchor*negative_example)
	return keras.losses.cosine_proximity(anchor,positive_example) - keras.losses.cosine_proximity(anchor,negative_example)



def loss_function(y_true,y_pred):
	#shape of y_pred at this point should be (batch_size,300)
	total_elements_in_batch = 16
	N = 8
	loss = tf.Variable(0.0)
	[first_half,second_half] = tf.split(y_pred,2,0)
	anchors = tf.split(first_half,8,0)
	examples = tf.split(second_half,8,0)

	for (i,anchor) in enumerate(anchors):
		temp_sum = tf.Variable(0.0,dtype='float32')
		for (j,example) in enumerate(examples):
			if i ==	j: continue
			temp_sum = tf.add(temp_sum,backend.exp(distance(anchor,examples[i],example)))
		temp_sum = tf.add(temp_sum,tf.constant(1,dtype='float32'))
		loss = tf.add(loss,tf.log(temp_sum))
	
	loss = tf.divide(loss,16)	
	print(tf.summary.scalar('sdfdsf', loss))
	return tf.convert_to_tensor([loss]*16)



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
opt2 = Dense(64, activation = 'relu')(opt)     #intialization?
model = Model(input = inp, output = opt2)

adam = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999)
sgd = optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False)
model.compile(optimizer=adam,loss=loss_function)

model.load_weights('n_pair.h5')



random_number = randint(1,3000)
live_image = train_data[0][random_number]
print("live images shape:",live_image.shape)
random_images = []
for i in range(0,8):
	random_images.append(train_data[i][randint(1,200)])

images = np.asarray(random_images)
print("images shape:",images.shape)

images = np.insert(images,8,live_image,axis=0)
print("images shape:",images.shape)



embeddings	= model.predict(images)
print(embeddings.shape)
print(embeddings)
# for i in range(0,8):
# 	print(np.linalg.norm(embeddings[8],embeddings[i]))0
	







