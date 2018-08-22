# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, 562].values
X_test = train.iloc[:, :-1].values
y_test = train.iloc[:, 562].values

#Exploring data
plt.hist(train['Activity'].map({'STANDING':2,'SITTING':1,'LAYING':0,'WALKING':3,'WALKING_DOWNSTAIRS':4,'WALKING_UPSTAIRS':5}))
plt.title('Activity (LAYING:0, SITTING:1, STANDING:2, WALKING:3,WALKING_DOWNSTAIRS:4,WALKING_UPSTAIRS:5)')
plt.show()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.fit_transform(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components =281)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components =None)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#predicting results
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn import metrics
accuracy0=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy0)

from sklearn.metrics import precision_recall_fscore_support
macro=precision_recall_fscore_support(y_test, y_pred, average='macro')
micro=precision_recall_fscore_support(y_test, y_pred, average='micro')
weight=precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(macro)
print(micro)
print(weight)