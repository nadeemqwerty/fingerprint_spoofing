import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,Lambda
from keras.layers import RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
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

def prepare_batch(train_data):
	anchors = []
	examples = []

	for class_ in train_data:
		random_pair = random.sample(class_,2)
		anchors.append(random_pair[0])
		examples.append(random_pair[1])
	# batch = anchors + examples
	# return np.asarray(batch)
	return [np.asarray(anchors),np.asarray(examples)]


def merge_function(x):
	[x1,x2] = x
	margin = 0
	x1_norm = K.l2_normalize(x1, axis=1)
	print("x1_norm shape:",x1_norm.get_shape())
	x2_norm = K.l2_normalize(x2, axis=1)

	neg_matrix = K.dot(x1_norm,K.transpose(x2_norm))
	print("neg_matrix shape:",neg_matrix.get_shape())
	pos_val = K.batch_dot(x1_norm,x2_norm,axes=1)
	print("pos_val shape:",pos_val.get_shape())
	#pos_matrix = tf.tile(pos_val,[1,tf.size(pos_val)])
	pos_matrix = RepeatVector(tf.size(pos_val))(pos_val)
	pos_matrix = K.squeeze(pos_matrix,2)
	print("pos_matrix shape:",pos_matrix.get_shape())
	return_value = K.mean(K.log(1 + K.sum(K.exp(neg_matrix - pos_matrix + margin),axis=1)),axis=0)
	print("Return value shape:",return_value.get_shape())
	return return_value
	


	ret = []
	for i in range(6):
		temp = 0
		for j in range(6):
			if(i==j):
				continue
			temp+=K.sqrt(K.sum(K.square(x2[j] - x1[j]), axis=-1, keepdims=True))
		ret.append(temp/5)
	return tf.convert_to_tensor(ret)

def merge_function_shape(x): 
	return (6,1)


def loss_function(y_true,y_pred):
	print("y_true shape:",y_true.get_shape())
	print("y_pred shape:",y_pred.get_shape())
	return K.reshape(K.abs(y_true-y_pred),shape=(6,1)) 

def my_model():
	anchors = Input(shape=(224,224,3))
	examples = Input(shape=(224,224,3))

	embedding_network_model = get_embedding_network() 
	anchors_embedding = embedding_network_model(anchors) 
	print(anchors_embedding.get_shape())
	examples_embedding = embedding_network_model(examples)
	print(examples_embedding.get_shape())
	
	merged = merge([anchors_embedding,examples_embedding],mode=merge_function,output_shape=merge_function_shape) 
	
	model = Model(inputs=[anchors,examples],outputs=merged)

	adam = optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999)
	sgd = optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False)

	model.compile(loss=loss_function,optimizer=adam)
	print(model.summary())

	return model


def get_embedding_network():
	model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
	vgg_model_output = model.layers[-1].output
	vgg_model_input = model.input
	x = Flatten()(vgg_model_output)
	x = Dense(4096, activation = 'relu')(x)     #intialization?
	x = Dense(4096, activation = 'relu')(x)
	x = Dense(128, activation = 'sigmoid')(x)
	model = Model(inputs = vgg_model_input,outputs = x)
	return model


print ("Loading 2013 biometric.`..")

dirs = ['live','Ecoflex','Gelatin','Latex','WoodGlue','Modasil']
#train_data is the list of arrays of images contained in respective folders in dirs list
train_data = []

dir_prefix = "../bio/"
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

# for i in model.layers:
#     weights = layer.get_weights()
iterations = 100
losses = []
min_losses = []
model = my_model()
k = model.layers[0].get_weights()
print('asdasda',k)

for i in range(1,iterations+1):
	print("iteration:",i)
	data = prepare_batch(train_data)
	#print(data)
	loss = model.train_on_batch(data,np.ones((6,1)))
	print(loss)	
	losses.append(loss)
	if i%50 == 0:
		min_losses.append(max(losses[-50:]))
	print(min_losses)
print(model.layers[0].get_weights())


model.save_weights('n_pair.h5')


#print(losses)
#check model accuracy.	



