import pandas as pd
import numpy as np
import matplotlib
import math
from sklearn.utils import shuffle

def read(file):
	train = pd.read_csv(file)
	s = 0
	data = train.dropna(subset=['song_hotttnesss'])
	#data = data[data.year != 0.0]
	data = data[data.song_hotttnesss != 0.0]
	return data

def next_batch(size,data):
	dsize = list(range(0,len(data)))
	batch = []
	for x in random.sample(size,dsize):
		features = []
		label = data.loc[x]['song_hotttnesss']

def test():
	a = read("MILLION_SONGS.csv")
	a = shuffle(a)
	#a.info()

	features = ['artist_hotttnesss','artist_familiarity','tempo','loudness','duration']
	hotness = a['artist_hotttnesss'].values
	hotness = hotness.reshape((-1,1))
	for x in features:
		values = (a[x].values).reshape((-1,1))
		hotness = np.concatenate((hotness,values),axis=1)
	labels = (a['song_hotttnesss'].values).reshape((-1,1))
	#print(hotness)
	hi = a['segments_timbre'].values
	#print(hi[0])

test()
#print(a['segments_timbre'])



