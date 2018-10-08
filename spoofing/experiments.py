import numpy as np
import tensorflow as tf

anchors = tf.constant(np.zeros((6,64))) 
examples = tf.constant(np.zeros((6,64)))

def get_embedding_network():
	model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
	vgg_model_output = model.layers[-1].output
	vgg_model_input = model.input
	x = tf.keras.layers.Flatten()(vgg_model_output)
	x = tf.keras.layers.Dense(4096, activation = 'relu')(x)     #intialization?
	x = tf.keras.layers.Dense(4096, activation = 'relu')(x)
	x = tf.keras.layers.Dense(128, activation = 'sigmoid')(x)
	model = tf.keras.models.Model(inputs = vgg_model_input,outputs = x)
	return model

def merge_function(x):
	[x1,x2] = x
	margin = 0
	x1_norm = tf.keras.backend.l2_normalize(x1, axis=1)
	x2_norm = tf.keras.backend.l2_normalize(x2, axis=1)

	neg_matrix = tf.keras.backend.dot(x1_norm,tf.keras.backend.transpose(x2_norm))
	pos_val = tf.keras.backend.batch_dot(x1_norm,x2_norm,axes=1)
	pos_matrix = tf.keras.layers.RepeatVector(tf.size(pos_val))(pos_val)
	pos_matrix = tf.keras.backend.squeeze(pos_matrix,2)
	return_value = tf.keras.backend.mean(tf.keras.backend.log(1 + tf.keras.backend.sum(tf.keras.backend.exp(neg_matrix - pos_matrix + margin),axis=1)),axis=0)
	
	return return_value

def loss_function(y_true,y_pred):
	return K.reshape(K.abs(y_true-y_pred),shape=(6,1)) 

def my_model():
	anchors = tf.keras.layers.Input(batch_shape=(None,224,224,3))
	examples = tf.keras.layers.Input(batch_shape=(None,224,224,3))

	embedding_network_model = get_embedding_network()
	anchors_embedding = embedding_network_model(anchors)
	examples_embedding = embedding_network_model(examples)
	
	merged = tf.keras.layers.merge([anchors_embedding,examples_embedding],mode=merge_function,output_shape=merge_function_shape) 
	
	model = tf.keras.layers.Model(inputs=[anchors,examples],outputs=merged)

	adam = tf.keras.optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999)
	sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False)

	model.compile(loss=loss_function,optimizer=adam)
	return model


model = my_model()

anchors = tf.placeholder(tf.float32,shape=(None,224,224,3))
examples = tf.placeholder(tf.float32,shape=(None,224,224,3))

final_value = model([anchors,examples])

sess = tf.Session()

writer = tf.summary.FileWriter('./experiment-tensorflow-logs')
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())
sess.run(final_value,feed_dict={anchors:np.zeros((6,224,224,3)),examples:np.zeros((6,224,224,3))})


