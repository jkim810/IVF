import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

import numpy as np
import time, os, sys
import pandas as pd
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl

SEED       = 42

TRAIN_SIZE = 0.65
VALID_SIZE = 0.15
TEST_SIZE  = 0.2

BATCH_SIZE = 32
EPOCHS     = 150

MASTER_FILE = 'data.csv'

# Load ImageNet class names
class2idx = {'EUP':0,
			 'ANU':1,
			 'CxA':2,
			 'MUT':3}

idx2class = {class2idx[k]:k for k in class2idx}


if __name__ == '__main__':
	# For Reproducability of code
	np.random.seed(SEED)

	# Torch Dataloader parameters
	params = {'batch_size': BATCH_SIZE,
			  'shuffle': True}
	
	# Read data
	df = pd.read_csv(MASTER_FILE)
	df = df[df['BX_DAY'] == 6]
	df = df[['PGD_RESULT', 'FILENAME']]
	
	# Plot Frequency of each elements
	# sns.countplot(x = 'PGD_RESULT', data = df)
	
	# Split dataset
	train_df, valid_df, test_df = np.split(df.sample(frac=1), [int(TRAIN_SIZE*len(df)), int((TRAIN_SIZE + VALID_SIZE)*len(df))])
	
	eup_train = train_df[train_df['PGD_RESULT'] == 'EUP'].head()
	anu_train = train_df[train_df['PGD_RESULT'] == 'ANU'].head()
	cxa_train = train_df[train_df['PGD_RESULT'] == 'CxA'].head()
	mut_train = train_df[train_df['PGD_RESULT'] == 'MUT'].head()

		
	imgs = [skimage.io.imread(f) for f in eup_train['FILENAME']]
	#imgs = [skimage]
	nimg = len(imgs)
	
	
	plt.figure(figsize=(12,4))
	for k,img in enumerate(imgs):

		plt.subplot(3,len(imgs),k+1)
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			left=False,
			labelleft=False,
			labelbottom=False)

		plt.imshow(img)
	
		plt.subplot(3,len(imgs),len(imgs)+k+1)

		# Load picture and detect edges
		image = img_as_ubyte(rgb2gray(img))
		edges = canny(image, sigma=3, low_threshold=10, high_threshold=20)
		plt.imshow(edges,cmap = plt.cm.gray)
		

		# Detect two radii
		hough_radii = np.arange(80, 150, 2)
		hough_res = hough_circle(edges, hough_radii)

		# Select the most prominent 3 circles
		accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
												   total_num_peaks=2)

		# Draw them
		image = color.gray2rgb(image)
		for center_y, center_x, radius in zip(cy, cx, radii):
			circy, circx = circle_perimeter(center_y, center_x, radius,
											shape=image.shape)
			image[circy, circx] = (255, 0, 0)

		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			left=False,
			labelleft=False,
			labelbottom=False)
		
		plt.subplot(3,len(imgs),2*len(imgs)+k+1)
		plt.imshow(image, cmap=plt.cm.gray)
	
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			left=False,
			labelleft=False,
			labelbottom=False)
	'''
	plt.figure(figsize=(12,4))
	sigma_values = [0.3, 1, 3, 10]

	for k,img in enumerate(imgs):
		plt.subplot(len(sigma_values) + 1,len(imgs),k+1)
		plt.imshow(img)
		plt.title('Original')
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			left=False,
			labelleft=False,
			labelbottom=False)
		
	for j, sig in enumerate(sigma_values):
		for k,img in enumerate(imgs):
			plt.subplot(len(sigma_values) + 1,len(imgs),(j+1)*len(imgs)+k+1)
			# Load picture and detect edges
			image = img_as_ubyte(rgb2gray(img))
			edges = canny(image, sigma=sig, low_threshold=10, high_threshold=20)
			plt.imshow(edges,cmap = plt.cm.gray)
			plt.title('Sigma:{}'.format(sig))
			plt.tick_params(
				axis='both',          # changes apply to the x-axis
				which='both',      # both major and minor ticks are affected
				bottom=False,      # ticks along the bottom edge are off
				top=False,         # ticks along the top edge are off
				left=False,
				labelleft=False,
				labelbottom=False)
	
	'''
	plt.show()
