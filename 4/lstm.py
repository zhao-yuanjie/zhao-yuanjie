from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM

from word2vec import *
from sklearn.metrics import classification_report
model=Sequential()
model.add(Embedding(len(dict)+1,256,input_length=50))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(X_train,y_train,batch_size=16,nb_epoch=10) #训练时间较长
print (classification_report(y_test, model.predict(X_test)))




