"""
This file implements a very simple knn benchmark for the didi competition.
"""

import didi_input
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split

# First, conduct a simple train_test_split
dataset = didi_input.load_data('public_data', 'training_data')

X_all = dataset.loc[:,['district_id','mean_price','time_slot','week_day']]
y_all = dataset.loc[:, 'gap']

# Make Week Day and Time Slot not to be zero.
X_all['time_slot'] = X_all['time_slot']+1
X_all['week_day'] = X_all['week_day']+1


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size=0.25,
                                                    random_state=100)
neigh = KNeighborsRegressor()
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

import ipdb; ipdb.set_trace()