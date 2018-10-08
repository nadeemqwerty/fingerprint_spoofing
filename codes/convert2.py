import matplotlib
matplotlib.use('TkAgg')
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
from scipy import ndimage,misc
import os.path
import cv2
from PIL import Image,ImageOps
import PIL

path3_ear = './live'
list_ear3 = os.listdir(path3_ear)
ear_samples3 = len(list_ear3)

for file in list_ear3:
	print (str(file))
	im = Image.open(path3_ear + '/' + file)
	im = im.convert('RGB')
	#im = ndimage.imread(path3_ear + '/' + file,flatten=False,mode = 'RGB')
	width, height = im.size
	left_i=0
	left_j=0
	right_i=0
	right_j=0
	up_i=0
	up_j=0
	down_i=0
	down_j=0

	try:
		for i in range(1,width):
			for j in range(1,height):
				r,g,b = im.getpixel((i,j))
				if(r<80 and g<80 and b<80):
				# if(r!=255 or g!=255 or b!=255):
					left_i = i
					left_j = j
					raise UnboundLocalError('My exit condition was met. Leaving try block')
	except UnboundLocalError:
		print ('Found')

	try:
		for i in range(width-1,1,-1):
			for j in range(1,height):
				r,g,b = im.getpixel((i,j))
				if(r<80 and g<80 and b<80):
				# if(r!=255 or g!=255 or b!=255):
					right_i=i
					right_j=j
					raise UnboundLocalError('My exit condition was met. Leaving try block')
	except UnboundLocalError:
		print ('Found')

	try:
		for j in range(1,height):
			for i in range(1,width):
				r,g,b = im.getpixel((i,j))
				if(r<80 and g<80 and b<80):
				# if(r!=255 or g!=255 or b!=255):
					up_i=i
					up_j=j
					raise UnboundLocalError('My exit condition was met. Leaving try block')
	except UnboundLocalError:
		print ('Found')

	try:
		for j in range(height-1,1,-1):
			for i in range(1,width):
				r,g,b = im.getpixel((i,j))
				if(r<80 and g<80 and b<80):
				# if(r!=255 or g!=255 or b!=255):
					down_i=i
					down_j=j
					raise UnboundLocalError('My exit condition was met. Leaving try block')
	except UnboundLocalError:
		print ('Found')

	im = im.crop((left_i,up_j,right_i,down_j))

	img = misc.imresize(im,(224,224))

	misc.imsave('livenew/' + str(file) , img)
	# misc.imsave('WoodGlueNew/' + str(file) , img)
	
	rot = ImageOps.mirror(im)
	rot = misc.imresize(rot,(224,224))
	# misc.imsave('WoodGlueNew/' + str(file) , rot)
	misc.imsave('livenew/' + str(file) , rot)

	rot = im.rotate(180)
	rot = misc.imresize(rot,(224,224))
	misc.imsave('livenew/' + 'C' + str(file) , rot)
	# misc.imsave('WoodGlueNew/' + 'C' + str(file) , rot)
	





