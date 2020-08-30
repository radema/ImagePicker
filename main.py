
from os import listdir, stat

from os.path import isfile, join

import shutil
from local_credentials import *



# files name in spotlight images folder
files = [f for f in listdir(spotlight_folder) if isfile(join(spotlight_folder, f)) 
	and stat(join(spotlight_folder, f)).st_size/(1024)>100]

#files name of images already present in Immagini
images = [f for f in listdir(destination_folder) if isfile(join(destination_folder, f))]

for file in files:
	target = join(destination_folder,file+'.jpg')
	#if file in images:
	shutil.copyfile(join(spotlight_folder,file), target)
	#shutil.copystat(join(spotlight_folder,file), target)