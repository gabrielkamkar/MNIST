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

os.chdir('C:\\Users\\kamka\\Documents\\Python\\Hands_On_ML')
######################################################################################################
### Question 1: MNIST classifier with over 97% accuracy on test set

train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

train = train.to_numpy(copy=False)
test = test.to_numpy(copy=False)

x_train, y_train, x_test, y_test = train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]

x_train.shape
x_test.shape

y_test = y_test.astype(np.uint8)
y_train = y_train.astype(np.uint8)
'''
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
# I think that that means that it predicts a one? right?

# there are sOOOOO many fucking ways to evaluate, should I do f1 score, precision, recall, anything else?

y_train_pred_1 = cross_val_predict(knn_clf, x_train, y_train, cv=3)

f1_score(y_train, y_train_pred_1, average='macro')
# 96.72%, so fucking close, lets get training score over 97 before we go to test set


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
### fuck!!!, 96.9% accuracy

# the best n_estimators parameter was 3, so lemme try 4 this time instead


knn_model_update_2 = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn_model_update_2.fit(x_train,y_train)

y_train_pred_3 = cross_val_predict(knn_model_update_2, x_train, y_train, cv=3)
f1_score(y_train, y_train_pred_3, average='macro')

## YES!!! 97% baby!

# now time to test it on the actual test data

final_predictions = knn_model_update_2.predict(x_test)
f1_score(y_test, final_predictions, average='macro')
# 0.97122

# DONE!

######################################################################################

### Question 2:

# start by making function
def shift_image(image, dx, dy):
    image = image.reshape((28,28))
    shifted_image = shift(image,[dx,dy],cval=0,mode='constant')
    return shifted_image.reshape([-1])

# make copy list of data
x_train_new = [image for image in x_train]
y_train_new = [label for label in y_train]

# make transformed copies and add to list
for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
    for image, label in zip(x_train, y_train):
        x_train_new.append(shift_image(image, dx, dy))
        y_train_new.append(label)

# turn transformed lists into np.arrays
x_train_new = np.array(x_train_new)
y_train_new = np.array(y_train_new)

# shuffle data so that the similar images arent near eachother
shuffle_indx = np.random.permutation(len(x_train_new))
x_train_new = x_train_new[shuffle_indx]
y_train_new = y_train_new[shuffle_indx]

# train new classifier
knn_new = KNeighborsClassifier(n_neighbors=4, weights = 'distance')
knn_new.fit(x_train_new,y_train_new)


# finding the score will take too long
y_pred = knn_new.predict(x_test)
f1_score(y_test,y_pred, score='macro')
# increases accuracy by around 0.5%!!
'''

