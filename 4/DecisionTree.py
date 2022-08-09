
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix,recall_score
from sklearn.metrics import classification_report
from word2vec import *
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
c_mat = confusion_matrix(y_test,y_pred)
kappa = cohen_kappa_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred,average='macro')
print("Confusion Matrix:\n", c_mat)
print("\nKappa: ",kappa)
print("\nAccuracy: ",acc)
print("\nrecall_score: ",recall)
print ('The Accuracy of RandomForest Classifier is:', model.score(X_test,y_test))
print (classification_report(y_test, model.predict(X_test)))