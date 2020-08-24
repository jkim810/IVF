import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import os
import shutil
from tqdm import tqdm

INPUT_DIR = '../data/'
OUTPUT_DIR = '../normalized_data/'

files = os.listdir(INPUT_DIR)

mean_array = []
std_array = []

# read / get statistics
for img_name in tqdm(files):
	img = Image.open(INPUT_DIR + img_name)
	np_img = np.array(img)
	img.close()

	mean_array.append(np.mean(np_img))
	std_array.append(np.std(np_img))


# save stats and filter dark/low contrast images
df = pd.DataFrame({'filename': files, 'mean': mean_array, 'std': std_array})
df['filter_val']  = df['mean'] + df['std']

print(df.sort_values('filter_val').head())
df.to_csv('../Stats/img_filter.csv', index = False)
df = df[df['filter_val'] > 82]

#plt.scatter(df['mean'], df['std'])
#plt.show()

'''
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# normalize images
for img_name in tqdm(df['filename'].values):
	
	img = Image.open(INPUT_DIR + img_name)
	np_img = np.array(img)
	img.close()

	cur_mean = np.mean(np_img)
	cur_std = np.std(np_img)

	np_img_normalized = (np_img - cur_mean) * global_std/cur_std + global_mean
	np_img_normalized = np_img_normalized.clip(min = 0, max = 255).astype(np.uint8)
	
	while np.abs(global_mean - cur_mean) > 5 and np.abs(global_std - cur_std) > 5:
		cur_mean = np.mean(np_img_normalized)
		cur_std = np.std(np_img_normalized)

		np_img_normalized = (np_img_normalized - cur_mean) * global_std/cur_std + global_mean
		np_img_normalized = np_img_normalized.clip(min = 0, max = 255).astype(np.uint8)

	img = Image.fromarray(np_img_normalized)
	img.save(OUTPUT_DIR + img_name)
'''