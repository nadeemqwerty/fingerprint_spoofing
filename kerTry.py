from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense
from keras.models import Model
import numpy as np
import random
import os
from PIL import Image
from keras import optimizers
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


def ms(a,b):
    return tf.square(tf.subtract(a,b))

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
                temp_s = tf.Variable(0.0,dtype='float32')
                for (j,example) in enumerate(examples):
                        if i == j: temp_s = ms(anchor, example)
                        temp_sum = tf.add(temp_sum,ms(anchor, example))
                temp_sum = tf.add(tf.divide(tf.negative(temp_sum),tf.constant(7,dtype='float32')),temp_s)
                loss = tf.add(loss,temp_sum)

        loss = tf.divide(loss,16)
        print(tf.summary.scalar('sdfdsf', loss))
        return tf.convert_to_tensor([loss]*16)






model  = VGG19(include_top = True, weights='imagenet', classes=1000)
model.layers.pop()

model.layers[-1].outbound_nodes = []
inp = model.input
opt = model.layers[-1].output
opt1 = Dense(1, activation = 'sigmoid', name='emb')(opt)
model = Model(input  = inp, output = opt1)

adam = optimizers.Adam(lr = 1e-6)

print(model.summary())
train_data = []
train_dir = './segmented/'
dirs = ['live','BodyDouble','Ecoflex','Gelatin','Latex','Playdoh','WoodGlue','Modasil']
for i in dirs:
    temp = []
    for file_name in os.listdir(train_dir+i):
        if file_name[-3:] == ".db":
            continue
        img = image.load_img(train_dir+i+"/"+file_name, target_size=(224,224,3))
        img_array = image.img_to_array(img)
        temp.append(img_array)
    temp = np.array(temp)
    temp.astype('float')/255.
    train_data.append(temp)


X_train = prepare_batch(train_data)
for i in range(1,200):
        X_train = np.concatenate((X_train,prepare_batch(train_data)),axis=0)
print(X_train.shape)

print("last layer's weights:")
print(model.layers[-1].get_weights())


model.compile(optimizer=adam,loss=loss_function)
model.fit(np.asarray(X_train), np.ones((16*200,300)), batch_size=16, nb_epoch=50,
          verbose=1, validation_split=0.2)

model.save_weights("n_pair_m.h5")
