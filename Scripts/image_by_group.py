import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from shutil import copyfile

scoredict = {
	'ICM':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
	'TE':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
	'BMS':[3.0, 6.0, 9.0,  12.0, 15.0],
	'BS':[3.0, 6.0, 9.0, 12.0, 15.0, 17.0],
	'Expansion':[1, 2, 3, 4, 5, 6]
}

df = pd.read_csv('../meta_numeric.csv')
filenames = os.listdir('../data')
filenames2id = {s[:8]:s for s in filenames}

df = df[df.columns[1:]]
df['ID'] = df['SUBJECT_NO'].astype(str)
df['ID'] = df['ID'].str.slice(stop=8)
df = df[df['ID'].isin(filenames2id.keys())]

df['FILENAMES'] = '../data/' + df['ID'].replace(filenames2id)


scores = ['ICM', 'TE', 'BMS', 'BS', 'Expansion']
for score in scores:
	subscores = sorted(df[score].dropna().unique())
	#subscores = subscores[~np.isnan(subscores)]
	print(score)
	print(scoredict[score])
	img_final = []
	out_filename = '../Preview/'+ str(score) + '.jpg'

	for subscore in reversed(scoredict[score]):
		tmp = df[df[score] == subscore].sample(n=5)
		in_filename = tmp['FILENAMES']
		images = [Image.open(in_file) for in_file in in_filename]
		img_array = np.hstack(images)
		img_final.append(img_array)

	img_final2 = np.vstack(img_final)
	if score == 'Expansion':
		plt.imshow(img_final2)
		plt.show()

	#im = Image.fromarray(img_final2)
	#im.save(out_filename)
			


