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
#import keras

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
	return backend.mean(backend.log(1 + backend.sum(backend.exp(neg_matrix - pos_matrix),axis=1)),axis=0)



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
# train_data = np.array(train_data)
# train_data = train_data.astype('float32')/255

model = VGG16(include_top=True, weights=None, classes=1000)
model.layers.pop()
model.layers[-1].outbound_nodes = []
inp = model.input
opt = model.layers[-1].output
opt2 = Dense(250, activation = 'sigmoid')(opt)     #intialization?
model = Model(input = inp, output = opt2)

adam = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999)
sgd = optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False)
model.compile(optimizer=adam,loss=loss_function)

iterations = 10000
losses = []
min_losses=[]

for i in range(1,iterations+1):
	print("iteration:",i)
	(anchors,examples) = prepare_batch(train_data)
	y_true = model.predict(examples)
	loss = model.train_on_batch(anchors,y_true)
	print(loss)	
	losses.append(loss)
	if i%50 == 0:
		min_losses.append(min(losses[-50:]))
	print(min_losses)


print("minimum:",min(losses))
print("average:",sum(losses)/len(losses))

# X_train = prepare_batch(train_data)
# for i in range(1,200):
# 	X_train = np.concatenate((X_train,prepare_batch(train_data)),axis=0)
# print(X_train.shape)

# print("last layer's weights:")
# print(model.layers[-1].get_weights())

# model.fit(np.asarray(X_train), np.ones((16*200,300)), batch_size=16, nb_epoch=50,
#           verbose=1, validation_split=0.2)


model.save_weights('n_pair.h5')


#print(losses)
#check model accuracy.	


