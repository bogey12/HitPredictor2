import numpy as np
import pdtest
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras import regularizers
from feature_process import parseseg
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.utils import np_utils

#np.set_printoptions(threshold=np.nan)
#from keras.datasets import cifar10

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

a = pdtest.read("MILLION_SONGS.csv")
features = ['artist_hotttnesss','artist_familiarity','tempo','loudness','duration']
hotness = a['artist_hotttnesss'].values
hotness = hotness.reshape((-1,1))
for x in features:
	values = (a[x].values).reshape((-1,1))
	hotness = np.concatenate((hotness,values),axis=1)

timbre = a['segments_pitches'].values
final = parseseg(timbre)
hotness = np.concatenate((hotness,final),axis=1)
labels = (a['song_hotttnesss'].values).reshape((-1,1))
labels = labels.tolist()

for i in range(0,len(labels)):
	for q in range(0,len(labels[i])):
		labels[i][q] = round(labels[i][q],1)

labels = np.array(labels)

x = []

for y in range(0,len(hotness)):
	temp = hotness[y].reshape(3,13,1)
	temp = temp.tolist()
	x = x + [temp]
x = np.array(x)

y_test = labels[0:844]
y_train = labels[844:len(labels)]
x_test = x[0:844]
x_train = x[844:len(x)]

model = Sequential()
model.add(Conv2D(39, kernel_size=3, padding='same', activation='relu', input_shape=(3,13,1,)))
model.add(Conv2D(39, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(39, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(39, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=32,nb_epoch=10,verbose=1)
train = model.evaluate(x_train,y_train,verbose=0)
score = model.evaluate(x_test,y_test,verbose=0)
print(history.history.keys())
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
print(score)
print(train)
#print(labels)
#print(hotness)

