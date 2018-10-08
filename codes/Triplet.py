import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
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

def Feature_Extractor(input_shape):
	input_img = Input(shape=(160,176,1))
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 80 x 88

	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 40 x 44

	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 20 x 22

	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 10 x 11

	conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)

	up6 = UpSampling2D((2,2))(conv5) # 20 x 22
	up6 = merge([up6, conv4], mode='concat', concat_axis=3)
	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)

	up7 = UpSampling2D((2,2))(conv6) # 40 x 44
	up7 = merge([up7, conv3], mode='concat', concat_axis=3)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)

	up8 = UpSampling2D((2,2))(conv7) # 80 x 88
	up8 = merge([up8, conv2], mode='concat', concat_axis=3)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)

	up9 = UpSampling2D((2,2))(conv8) # 160 x 176
	up9 = merge([up9, conv1], mode='concat', concat_axis=3)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)

	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9)
	ada=Adadelta(lr=5.0, rho=0.95, epsilon=1e-08, decay=0.001)
	rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)

	autoencoder = Model(input_img, decoded)
	#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.compile(loss='mean_squared_error', optimizer=rms)
	
	
	autoencoder.load_weights("Live1.h5")
	for i in range(27):
		autoencoder.layers.pop()
	for layers in autoencoder.layers:
		layers.trainable = False
	autoencoder.summary()
	return autoencoder
'''def Loss_NW(input_shape):
	left_input = Input(input_shape)
	right_input = Input(input_shape)
	input_combined = [left_input,right_input]
	y_correlated = Normalized_Correlation_Layer()([left_input,right_input])
	outp = Conv2D(128,(3,3),activation='relu',padding='same')(y_correlated)
	outp=MaxPooling2D()(outp)
	outp = Conv2D(256,(3,3),activation='relu',padding='same')(outp)
	outp=Flatten()(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(2000, activation = 'relu')(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(1000, activation = 'relu')(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(100, activation = 'relu')(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(1, activation = 'sigmoid')(outp)
	Network=Model(input=input_combined, output = outp)
	return Network'''
	
def Loss_NW(input_shape):
	left_input = Input(input_shape)
	right_input = Input(input_shape)
	input_combined = [left_input,right_input]
	convnet = Sequential()
	convnet.add(Flatten(input_shape = input_shape))
	#convnet.add(Dropout(0.3))
	#convnet.add(Dense(2048,activation="sigmoid"))
	encoded_l = convnet(left_input)
	encoded_r = convnet(right_input)
	#merge two encoded inputs with the l1 distance between them
	L1_distance = lambda x: K.abs(x[0]-x[1])
	#May think of another distance
	both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
	both = Dropout(0.3)(both)
	prediction=Dense(1000,activation='relu')(both)
	prediction = Dropout(0.3)(prediction)
	prediction = Dense(1, activation = 'sigmoid')(prediction)


	Network=Model(input=input_combined, output = prediction)
	return Network	
	
input_shape = (160, 176, 1)
Cnn_network = Feature_Extractor(input_shape)
#Cnn_network.load_weights("../Models/CNN_1.h5")
middle_shape = (10,  11, 256)
L_network = Loss_NW(middle_shape)
#L_network.load_weights("../Models/Loss_NW_10_10.h5")
#L_network.summary()
left_input = Input(input_shape)
middle_input = Input(input_shape)
right_input = Input(input_shape)
input_combined = [left_input,middle_input,right_input]

encoded_l = Cnn_network(left_input)
encoded_m = Cnn_network(middle_input)
encoded_r = Cnn_network(right_input)

result1 = L_network([encoded_l, encoded_m])
result2 = L_network([encoded_m, encoded_r])

L1_distance = lambda x: (x[1]-x[0])
both = merge([result1, result2], mode = L1_distance, output_shape=lambda x: x[0])
L_network.load_weights("Loss_Triplet_new_10_10.h5")
Cnn_network.load_weights("CNN_Triplet_new_10_10.h5")
Final_network = Model(input = input_combined, output = both)
Final_network.summary()
#exit()


#lr = 0.00006
lr = 0.001
optimizer = Adam(lr)
Final_network.compile(loss="mean_squared_error",optimizer=optimizer)

print 'Loading Data...............'
images=[]

live=os.listdir(os.getcwd()+"/Live_new")
labels=[]
for file in live:
	img=array(Image.open(os.getcwd()+"/Live_new"+"/"+file).convert('L')).reshape((160,176,1))
	images.append(img)
	labels.append(0)
	
fake=os.listdir(os.getcwd()+"/Fake_new")
for folder in fake:
	files=os.listdir(os.getcwd()+"/Fake_new/"+folder)
	for file in files:
		img=array(Image.open(os.getcwd()+"/Fake_new/"+folder+"/"+file).convert('L')).reshape((160,176,1))
		images.append(img)
		labels.append(1)

images=np.array(images)
print(images.shape)
images = images.astype('float32')/ 255.

print '----------------------Data Loaded------------------------------'
		
		
def make_batch(s1,p1,n):

    # both s1 and p1 are 1-based indexed.
    # pair[0] is positive, pair[2] is negative
    # pair[0] is positive from p = 0 to p = 4, both inclusive.
    # modify image-shape, assumes (1,192,160,1) shape.

    #live - 3750
    #fake - 3000

    pairs = [np.zeros((n, 160, 176, 1)) for i in range(3)]
    targets = np.ones((n,))

    pairs[0][0,:,:,:] = images[(s1-1)*3000+p1-1]
    pairs[1][0,:,:,:] = images[(s1-1)*3000+p1-1]
    if(s1==1):
    	s_negative = 1
    else:
    	s_negative = 0
    p_negative = rng.randint(3000)
    pairs[2][0,:,:,:] = images[s_negative*3000+p_negative]

    for i in range(1,n):
    	p_positive = rng.randint(3000)
        pairs[0][i,:,:,:] = images[(s1-1)*3000+p_positive]
        pairs[1][i,:,:,:] = images[(s1-1)*3000+p1-1]
        if(s1==1):
	    	s_negative = 1
        else:
	    	s_negative = 0
        p_negative = rng.randint(3000)
        pairs[2][i,:,:,:] = images[s_negative*3000+p_negative]

    return pairs,targets

	
Iterations = 2000
N = 64
output_file_name = "Triplet_NW_10_10.txt"
	
for i in range(1,Iterations+1):
	print('Iteration '+str(i)+' started')
	# loss = [0 for x in range(3)]
	train_loss = 0.0
	for s in range(1,2):
		if(s==1):											###387
			for p in range(1,3000):										##11
				pairs,label = make_batch(s,p,N)
				current_loss = Final_network.train_on_batch(pairs,label)
				train_loss += current_loss
				print ("Current Loss of " + str(p) + " : " + str(current_loss))
		else:
			for p in range(1,3000):										##11
				pairs,label = make_batch(s,p,N)
				current_loss = Final_network.train_on_batch(pairs,label)
				train_loss += current_loss
				print ("Current Loss of " + str(p) + " : " + str(current_loss))
		print ("Train Loss: " + str(current_loss))
	L_network.save_weights("Loss_Triplet_new_10_10.h5")
	Cnn_network.save_weights("CNN_Triplet_new_10_10.h5")
	print 'Weights Saved'
    # take first 10 subjects 5th pose, match with 1-4 of first 25, save loss
	"""test_loss = 0.0
	for ijk in xrange(10):
		probe_image = Cnn_network.predict(images[ijk*20+18].reshape(1, 128, 128, 1))
		for j in xrange(10):
			for k in xrange(15):
				gallery_image = Cnn_network.predict(images[j*20+k].reshape(1, 128, 128, 1))
				score = L_network.predict([probe_image,gallery_image])
				if j==ijk: test_loss += score
				else: test_loss += (1.0-score)

    # test end 
	print 'iteration ' + str(i) + ' training loss: ' + str(train_loss) + ' test_loss: ' + str(test_loss[0][0])
	with open(output_file_name,'a+') as f:
		f.write('Iteration ' + str(i) + '\ttraining loss ' + str(train_loss) + '\ttest_loss ' + str(test_loss[0][0]) + '\n')"""

