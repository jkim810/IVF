'''
GRADE_map = {
 'MOR',
 'CAV M',
 'Arr',
 '6CC',
 '6CB-/C',
 '6CB-',
 '6CB',
 '6CA-',
 '6B-C',
 '6B-B-/C',
 '6BB-/C',
 '6B-B-',
 '6B-B',
 '6BB-',
 '6BB',
 '6B-A-',
 '6BA-',
 '6B-/CC',
 '6B-/CB-/C',
 '6B-/CB-',
 '6B-/CB',
 '6A-B-/C',
 '6A-B-',
 '6A-B',
 '6AB-',
 '6AB',
 '5CC',
 '5CB-',
 '5CB',
 '5B-B-/C',
 '5B-B-',
 '5B-B',
 '5BB-',
 '5BB',
 '5B-/CB-'
 '5A-B-'
 '5A-A-'
 '4CC'
 '4CB-/C'
 '4CB-'
 '4CB'
 '4B-B-/C'
 '4BB-/C'
 '4B-B-'
 '4B-B'
 '4BB-'
 '4BB'
 '4B-A-'
 '4BA-'
 '4B-/CC'
 '4B-/CB-/C'
 '4B-/CB-'
 '4B-/CB'
 '4A-B-/C'
 '4A-B'
 '4AB-'
 '4AB'
 '3CC'
 '3CB-/C'
 '3CB-'
 '3CB'
 '3B-C'
 '3BC'
 '3B-B-/C'
 '3BB-/C'
 '3B-B-'
 '3B-B'
 '3BB-'
 '3BB'
 '3B-A-'
 '3B-A'
 '3BA-'
 '3BA'
 '3B-/CC'
 '3B-/CB-/C'
 '3B-/CB-'
 '3B-/CB'
 '3B-/CA-'
 '3A-B-/C'
 '3AB-/C'
 '3A-B-'
 '3A-B'
 '3AB-'
 '3AB'
 '3A-A-'
 '3A-A'
 '3AA-'
 '3AA'
 '2CC'
 '2CB-/C'
 '2CB-'
 '2CB'
 '2B-C'
 '2BC'
 '2B-B-/C'
 '2BB-/C'
 '2B-B-'
 '2B-B'
 '2BB-'
 '2BB'
 '2B-A'
 '2BA-'
 '2B-/CC'
 '2B-/CB-/C'
 '2B-/CB-'
 '2B-/CB'
 '2A-C'
 '2A-B-/C'
 '2AB-/C'
 '2A-B-'
 '2A-B'
 '2AB-'
 '2AB'
 '2A-A-'
 '2A-A'
 '2AA-'
 '2AA'
 '2-3CC'
 '2-3CB-/C'
 '2-3CB-'
 '2-3CB'
 '2-3B-C'
 '2-3BC'
 '2-3B-B-/C'
 '2-3BB-/C'
 '2-3BB-/c'
 '2-3B-B-'
 '2-3B-B'
 '2-3BB-'
 '2-3BB'
 '2-3BA-'
 '2-3B-/CC'
 '2-3B-/CB-/C'
 '2-3B-/CB-'
 '2-3B-/CB'
 '2-3AC'
 '2-3AB-/C'
 '2-3A-B-'
 '2-3A-B'
 '2-3AB-'
 '2-3AB'
 '2-3A-A-'
 '2-3A-A'
 '2-3AA-'
 '2-3AA'
 '1CC'
 '1CB-/C'
 '1CB-'
 '1CB'
 '1B-C'
 '1BC'
 '1B-B-/C'
 '1BB-/C'
 '1B-B-'
 '1B-B'
 '1BB-'
 '1BB'
 '1B-/CC'
 '1B-/CB-/C'
 '1B-/CB-'
 '1B-/CB'
 '1A-B-'
 '1A-B'
 '1-2CC'
 '1-2CB-/C'
 '1-2CB-'
 '1-2B-C'
 '1-2BC'
 '1-2B-B-/C'
 '1-2BB-/C'
 '1-2B-B-'
 '1-2B-B'
 '1-2BB-'
 '1-2BB'
 '1-2B-/CC'
 '1-2B-/CB-/C'
 '1-2B-/CB-'
 '1-2B-/CB'
 '1-2A-B-/C'
 '1-2A-B-'
 '1-2A-B'
 '1-2AB'
 '1-2A-A-'
}
'''

import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_excel('../Metadata.xlsx')

#print(df['Expansion'].unique())
print(df['ICM'].unique())
print(df['TE'].unique())

ICM_map = {
	'C' : 5,
	'B-/C' : 4,
	'B-' : 3,
	'B' : 2,
	'A-' : 1,
	'A' : 0,
	'N' : None
}

TE_map = {
	'C' : 5,
	'B-/C' : 4,
	'B-' : 3,
	'B' : 2,
	'A-' : 1,
	'A' : 0,
	'N' : None,
	'CM' : None
}


df['ICM'] = df['ICM'].replace(ICM_map)
df['TE'] = df['TE'].replace(TE_map)

df[['ICM', 'TE']] = df[['ICM', 'TE']].astype(float)
'''
pca = PCA()
pca.fit(df[['Expansion', 'ICM', 'TE']])
xfrm = pca.transform(df[['Expansion', 'ICM', 'TE']])
print(xfrm)
'''
df.to_csv('../meta_numeric.csv')