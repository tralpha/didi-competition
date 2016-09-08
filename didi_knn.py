"""
This file implements a very simple knn benchmark for the didi competition.
"""

# import didi_input
# import didi_test
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split
import cPickle as pkl
import pandas as pd

# dataset = didi_input.load_data('public_data', 'training_data')
### Load the Training Set
with open('public_data/training_data/clean_train.pickle', 'rb') as f:
    dataset = pkl.load(f)

dataset.reset_index(inplace=True)

X_all = dataset.loc[:,dataset.columns != 'gap']
y_all = dataset.loc[:, 'gap']


# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
#                                                     test_size=0.25,
#                                                     random_state=10)

with open('public_data/test_set_1/clean_test_1.pickle', 'rb') as t:
    test_set = pkl.load(t)

test_time_date = test_set.loc[:,['Time']]
test_set.drop(['Time','date'], axis=1, inplace=True)

# Rename the features which were not properly named
test_set.rename(columns={'second_order':'second_orders', 
                         'third_order':'third_orders',
                         'first_order':'first_orders'}, inplace=True)

test_set = test_set[X_all.columns]

import ipdb; ipdb.set_trace()

neigh = KNeighborsRegressor()
neigh.fit(X_all, y_all)

y_pred = neigh.predict(test_set)



final_csv = pd.concat([test_set.loc[:,['district_id']],test_time_date],axis=1)

final_csv.loc[:,'Prediction'] = y_pred

final_csv.loc[:,'district_id'] = final_csv.loc[:,'district_id'].astype(int)

final_csv.to_csv('ralph.csv', header=False, index=False)


# mae = mean_absolute_error(y_test, y_pred)
# print mae


