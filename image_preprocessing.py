from zipfile import ZipFile 
import numpy as np
import pydicom
import os
import cv2
import pandas as pd
from PIL import Image
# assuming you the files as downloaded from NDA
# run this script in the folder where you have all the followup folders (i.e. P001, 00m, etc.)

def resize(img, target_minsize):
	if min(img.shape) > target_minsize:
		y = img.shape[0] # y, rows
		x = img.shape[1] # x, cols
		if x > y:
			# too wide
			size = int(y / x * target_minsize)
			img = cv2.resize(img, (target_minsize, size), interpolation = cv2.INTER_AREA)
		elif x < y:
			# too tall
			size = int(x / y * target_minsize)
			img = cv2.resize(img, (size, target_minsize), interpolation = cv2.INTER_AREA)
		else:
			img = cv2.resize(img, (target_minsize, target_minsize), interpolation = cv2.INTER_AREA)
	return img

def inverting_bbox_crop(img):
	# first do homogenous crop (no equalization)
	change = True
	while change:
		size = img.shape[0] * img.shape[1]
		
		# top bottom crop
		temp = img.copy()
		temp[img.std(axis=1) < 3] = 0
		
		img = np.array(Image.fromarray(img).crop(Image.fromarray(temp).getbbox()))
		
		# left right crop
		temp = img.copy()
		temp[:,img.std(axis=0) < 3] = 0
		
		img = np.array(Image.fromarray(img).crop(Image.fromarray(temp).getbbox()))
		
		change = size != (img.shape[0] * img.shape[1])
	
	img = cv2.equalizeHist(img)
	# next do thresholding crop (with equalization)
	change = True
	while change:
		size = img.shape[0] * img.shape[1]
		
		img = Image.fromarray(img)
		img = np.array(img.crop(img.getbbox()))
		img = 255 - img
		img = cv2.equalizeHist(img)
		
		img = Image.fromarray(img)
		img = np.array(img.crop(img.getbbox()))
		img = 255 - img
		img = cv2.equalizeHist(img)
		
		change = size != (img.shape[0] * img.shape[1])
	return img


OUTDIR = 'image_preprocessing_out'
os.makedirs(OUTDIR, exist_ok=True)

archives = ['P001','00m','12m','24m','36m','48m','72m','96m']

for archive in archives:
	print('')
	print(archive)
	with ZipFile('{}/results/{}.zip'.format(archive,archive), 'r') as zf:
		filelist = [f.replace('_1x1.jpg','/001') for f in zf.namelist() if f.endswith('_1x1.jpg')]
		for i,file in enumerate(filelist):
			if os.path.exists('{}/{}_1x1.jpg'.format(OUTDIR,file.split('/')[-2])):
				continue
			print(i,'/',len(filelist), end='\r')
			try:
				with zf.open(file) as f:
					dcm = pydicom.dcmread(f)
					if dcm.Modality == 'MR':
						continue
					img = dcm.pixel_array
			except KeyError:
				print('file not found:', file)
				continue
			img = resize(img, 1024)
			qmax = np.quantile(img,0.99)
			qmin = np.quantile(img,0.1)
			qrange = qmax-qmin
			img = np.clip(((img-qmin)/qmax * 255),0,255).astype(np.uint8)
			if dcm.PhotometricInterpretation.endswith('1'):
				img = 255 - img
			img = inverting_bbox_crop(img)
			img = resize(img, 512)
			assert cv2.imwrite('{}/{}_1x1.jpg'.format(OUTDIR,file.split('/')[-2]), img)