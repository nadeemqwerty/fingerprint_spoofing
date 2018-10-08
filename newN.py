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

def print_anchors(anchors):
	for anchor in anchors:
		print(backend.eval(anchor))

def prepare_batch(train_data):
	anchors = []
	examples = []

	for class_ in train_data:
		random_pair = random.sample(class_,2)
		anchors.append(random_pair[0])
		examples.append(random_pair[1])
	batch = anchors + examples
	return np.asarray(batch)




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

for list_ in train_data:
	print(len(list_))

model = VGG16(include_top=False, weights='imagenet')
model.summary()
inp = model.input
output_vgg16_conv = model_vgg16_conv(inp)
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(input=inp,output=x)
print(model.summary())

adam = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999)
sgd = optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False)
model.compile(optimizer=adam,loss=loss_function)

iterations = 1000
losses = []

# for i in range(1,iterations+1):
# 	print("iteration:",i)
# 	data = prepare_batch(train_data)
# 	print(data.shape)
# 	loss = model.train_on_batch(data,np.ones((16,300)))
# 	print(loss)
# 	losses.append(loss)

X_train = prepare_batch(train_data)
for i in range(1,200):
	X_train = np.concatenate((X_train,prepare_batch(train_data)),axis=0)
print(X_train.shape)

print("last layer's weights:")
print(model.layers[-1].get_weights())

model.fit(np.asarray(X_train), np.ones((16*200,300)), batch_size=16, nb_epoch=50,
          verbose=1, validation_split=0.2)


model.save_weights('n_pair.h5')


#print(losses)
#check model accuracy.
