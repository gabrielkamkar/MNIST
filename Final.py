import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from scipy.ndimage.interpolation import shift


train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

train = train.to_numpy(copy=False)
test = test.to_numpy(copy=False)

x_train, y_train, x_test, y_test = train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]

x_train.shape
x_test.shape

y_test = y_test.astype(np.uint8)
y_train = y_train.astype(np.uint8)


# graph a number to see if it worked
y_train[0]

test_number = x_train[0].reshape(28, 28)
plt.imshow(test_number, cmap=mpl.cm.binary, interpolation='nearest')
plt.show()
# that's a five alright

# lets start with a regular knn classifier and see how well it does

some_digit = x_train[3] # this is a 1

knn_clf = KNeighborsClassifier()
knn_clf.fit(X = x_train, y = y_train)

knn_clf.predict([some_digit])

####

y_train_pred_1 = cross_val_predict(knn_clf, x_train, y_train, cv=3)

f1_score(y_train, y_train_pred_1, average='macro')
# 96.72%


# lets start with grid search

param_grid = [
    {'n_neighbors': [3,5,9,11,15], 'weights': ['uniform','distance']}
]

knn_model = KNeighborsClassifier()

grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                           return_train_score= True)

grid_search.fit(x_train, y_train)

grid_search.best_params_
grid_search.best_estimator_

knn_model_update = grid_search.best_estimator_

y_train_pred_2 = cross_val_predict(knn_model_update, x_train, y_train, cv=3)

f1_score(y_train, y_train_pred_2, average='macro')
### 96.9% accuracy

# the best n_estimators parameter was 3, so lemme try 4 this time instead


knn_model_update_2 = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn_model_update_2.fit(x_train,y_train)

y_train_pred_3 = cross_val_predict(knn_model_update_2, x_train, y_train, cv=3)
f1_score(y_train, y_train_pred_3, average='macro')

## 97% 

# now time to test it on the actual test data

final_predictions = knn_model_update_2.predict(x_test)
f1_score(y_test, final_predictions, average='macro')
# 0.97122

