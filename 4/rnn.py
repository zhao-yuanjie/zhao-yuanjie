
from sklearn.metrics import classification_report
from word2vec import *
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
clf.fit(X_train, y_train)
print ('The Accuracy of bp Classifier is:', clf.score(X_test, y_test))
print(classification_report(y_test,clf.predict(X_test)))