import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_data = pickle.load(open('data/data_batch_1', 'rb'),encoding='latin1')
X_train = np.array(train_data['data'])
y_train = np.array(train_data['labels'])
test_data = pickle.load(open('data/test_batch', 'rb'),encoding='latin1')
X_test = np.array(test_data['data'][:1000])
y_test = np.array(test_data['labels'][:1000])

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print('fitting')
## four different kernels for Support Vector Machine
clf = SVC(kernel='sigmoid', C=1.0, gamma=0.00033)
# clf = SVC(kernel='rbf', C=1.0, gamma=0.00033)
# clf = SVC(kernel='linear', C=1.0, gamma=0.01)
# clf = SVC(kernel='poly', C=1.0, gamma=0.01)
clf.fit(X_train_std, y_train)

y_pred = clf.predict(X_test_std)

print ("Accuracy: ", accuracy_score(y_test,y_pred))