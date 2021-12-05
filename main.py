from flask import Flask, render_template, request
import pandas as pd
import joblib
from PIL import Image
import numpy as np
import urllib.request
import urllib.parse

import keras
from keras.models import load_model
from keras.preprocessing import image as im

from skimage import color
from skimage import io

import cv2

import pytesseract

import tensorflow as tf
global graph,model2,autoencoder
graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)
model2 = load_model('final.h5')    #Lenet-aw.h5

autoencoder = load_model('happyHack.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get():
	input = dict(request.form)


	print("filename=",request.form.get('imgname'))  

	print("filename=",input['imgname'][0])   

	v= request.form.get('imgname')
	

	data = np.array([np.array(Image.open(v))])  #f.filename
	i=cv2.imread(v,0); # input as grey scale
	width = 256
	height = 256
	dim = (width, height)

	resized = cv2.resize(i, dim, interpolation = cv2.INTER_AREA)

	data = np.array([np.array(resized)])

	data = data.reshape(-1,256,256,1)
	data = data.astype('float32')
	data = data/ 255.

	y = model2.predict(data, verbose=1)

	print(" Y =  ",y)
	result = np.where(y[0] == np.amax(y[0]))
	print("result[0]=",result[0])
	print("result[0][0]=",result[0][0])
	pre=result[0][0]

	img2 = cv2.imread(v, 0)
	# cropped=img2[115:130, 30:100]
	if pre == 0:
	    cropped=img2[115:130, 30:100]     # 10-sns (30,115) (100,130)
	elif pre == 1:
	    cropped=img2[250:300, 70:265]     # 20-sns (70,250) (265,300)
	elif pre == 2:    
	    cropped=img2[450:520, 130:430] #100-sns
	elif pre == 3:    
	    cropped = img2[150:200, 75:390] #500-sns    
	# cropped = img2[180:250, 80:550] #realmoney (75,150)(390 ,200)

	config = ("-l eng --oem 3 --psm 6")
	number = cv2.fastNlMeansDenoising(cropped, 10, 7,21)
	number[number > 170] = 255
	number[number <= 170] = 0


	#text = pytesseract.image_to_string(number, config=config)
	text = '1'
	print("text=",text)

	xtest=cv2.imread(v, 0)
	xtest = cv2.resize(xtest,(256,256))
	xtest=xtest.reshape(-1,256,256,1)
	xtest = xtest.astype('float32')
	xtest = xtest/ 255.

	decimg = autoencoder.predict(data, verbose=1)

	mse1 = np.mean(np.power(xtest - decimg, 2),axis=1)
	mse0 = np.mean(mse1, axis=1)
	print(mse0[0])
	res=mse0[0]
	con=0
	if res > 0.002 :
		if res < 0.01:
			con=1

	if con == 0:
		return "-0-----------";	
	ret = str(result[0][0]) + str("1") + text;
	print("ret= ",ret);
	return ret;
    
#if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True, port=4000)
