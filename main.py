
from collections import Counter
from os import listdir, stat
from os.path import isfile, join
import PIL.Image as Image
import numpy as np

import shutil
from local_credentials import *

import tensorflow as tf
from tensorflow.keras import models

model = models.load_model(model_folder)
print("Model Loaded!")

print("Reading images...")
# files name in spotlight images folder
files = [f for f in listdir(spotlight_folder) if isfile(join(spotlight_folder, f)) 
	and stat(join(spotlight_folder, f)).st_size/(1024)>100]

#files name of images already present in Immagini
images = [f for f in listdir(destination_folder) if isfile(join(destination_folder, f))]

counter= Counter()

for file in files:
	target = join(destination_folder,file+'.jpg')
	im = Image.open(join(spotlight_folder,file))
	w,h = im.size
	# we move only not cropped images
	if w>h:
		print("Detected image")
		im = np.array(im.resize((224,224)))/255.0
		result = model.predict(im[np.newaxis, ...])[0,0]
		if 0.34 < result <0.76:
			shutil.copyfile(join(spotlight_folder,file), target)
			counter.update(["uncertainty"])
		elif result >= 0.76:
			counter.update(['good'])
			#print("Test!")
			shutil.copyfile(join(spotlight_folder,file),join(picture_folder,file+'.jpg') )
		else:
			counter.update(['bad'])
print("\n")
print(counter)
print("\n")