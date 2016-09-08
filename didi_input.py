"""
A simple python file to preprocess the data for the Didi competition
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os
import datetime
import cPickle as pkl


PREVIOUS = {'first': 1, 'second': 2, 'third': 3}


def get_week_day(date):
    """
    A function to get the week day of a particular date. I'll simply use the 
    actual week day of the dates to obtain the week days for the different 
    order transactions
    Args:
        date: The date of the particular order being processed
    Returns:
        week_day: The week day of that particular order being processed
    """
    date_datetime = datetime.datetime.strptime(date, '%Y-%m-%d')
    week_day = date_datetime.weekday() + 1
    return week_day


def load_test(test_path, test_set, force=False):
    """
    This function loads the test set and puts it in a dataframe which then
    preprocesses the test set and puts it in a dataframe with the following 
    features:
    1. District_id
    2. Time Slot
    3. Day of the Week

    Args:
        test_path: The path to the whole dataset
        force: A parameter to justify whether to force the loading of the 
               test set or not.
    Returns:
        test_result, test_set: A tuple, consisting of two pandas objects, 
                               which are the test_results and the test_set.
    """
    test_set_path = os.path.join(test_path, test_set)
    test_set = pd.read_table(test_set_path, header=None, delimiter=',',
                                names=['Time'], skiprows=1)

    test_times = pd.DataFrame(test_set.Time.str.rsplit('-',1).tolist(),
                              columns=['date', 'time_slot'])
    test_set = pd.concat([test_set, test_times], axis=1)
    test_set['week_day'] = test_set['date'].apply(get_week_day)
    test_set['time_slot'] = pd.to_numeric(test_set['time_slot'])
    full_test_set = pd.DataFrame(columns=['Time', 'date', 'time_slot',
                                          'week_day', 'district_id'])
    for i in range(1,67):
        test_set['district_id'] = [i]*len(test_set)
        full_test_set=pd.concat([full_test_set, test_set], ignore_index=True)
    # import ipdb; ipdb.set_trace()
    full_test_set.set_index(['district_id','time_slot'],inplace=True)
    return full_test_set


# A Global Dictionary Variable for Convenience to load the test set
print('Loading Test Sets...')
TEST_SETS = {'test_set_1': load_test('public_data','test_set_1/read_me_1.txt'),
             'test_set_2': load_test('public_data','test_set_2/read_me_2.txt')}
print('Test Sets Loaded :)')


def load_data(path, dataset, force=False):
    """
    Loads the data and returns it into a format which can be worked on by 
    a simple regression algorithm
    Args:
        path: A String, which indicates the path of the dataset to be loaded.
        dataset: A String, which indicates whether the dataset is a training 
                 set or the test set.
        force: A boolean value which indicates on whether to force the data 
               loading or not
    Returns:
        data: The dataset in the form of a Pandas Table to be loaded into the
              algorithm
    """
    dataset_path = os.path.join(path, dataset)
    # orders_path = os.path.join(dataset_path, 'order_data')
    # print(orders_path)
    data = process_orders(dataset_path, dataset, force)
    return data


def create_slots():
    """
    A simple function to create the time slots used by Didi
    Returns:
        time_slots: A List of dictionaries, which indicates the different 
                    starting and ending points of the different timeslots.
                    The Dic has two keys: start and end. Start marks the 
                    beginning of the time slot, while end marks the end 
                    of the time slot. The time for start and end are in 
                    the form of a datetime.datetime object, where the year is 
                    1900, month is 01, and day is 01. Only the time matters in 
                    this case.
    """
    time_slots = []
    start = datetime.datetime.strptime('00:00:00','%H:%M:%S')
    for i in range(1,145):
        slot_start = start + datetime.timedelta(minutes=10*(i-1))
        slot_end = (start + datetime.timedelta(minutes=10*i) - 
                    datetime.timedelta(seconds=1))
        time_slots.append({'start': slot_start, 
                           'end'  : slot_end})
    return time_slots


def allocate_slots(timestamp, time_slots):
    """
    This function takes a particular time string, and outputs which particular
    time_slot it belongs to.
    Args:
        timestamp: The timestamp of the particular order id.
    Return:
        time_slot: The timeslot of the particular order id.

    TODO: Due to the usage of apply(), this function makes te feature
          extraction very very slow. Any one can think of a way to make it 
          faster? :)
    """
    timestamp_datetime = datetime.datetime.strptime(timestamp[11:],'%H:%M:%S')
    assign_slot = 0
    for slot in time_slots:
        if (slot['start'] <= timestamp_datetime and 
            slot['end'] >= timestamp_datetime):
            time_slot = time_slots.index(slot) + 1
            assign_slot += 1
    if assign_slot != 1: # Check if the slot was assigned twice for an order_id
        import ipdb; ipdb.set_trace()
    return time_slot


def hash_to_id(district_hash, hash_table):
    """
    A function which changes the district_hash to the corresponding district
    id.
    Args:
        district_hash: The District_hash coming from the table
    Returns:
        district_id: The District_id coming from the cluster_map file
    """
    district_id = int(hash_table[hash_table['Hash'] == district_hash]['ID'])
    # import ipdb; ipdb.set_trace()
    return district_id


def train_last_three(train_set, feature, feature_name):
    """
    A function which is used as a helper function to the train_feats function
    below. It adds features to the train set in a similar manner as features 
    were also added to the testing set.
    Args:
        train_set: The training set to which the features are added
        feature: The feature to be added to the train set for the last three
                 transactions
    Returns:
        train_set: The train set with the added features
    """
    for pre in PREVIOUS:
        train_set.loc[:,pre+'_'+feature_name] = feature.shift(PREVIOUS[pre])
        train_set.loc[:,pre+'_'+feature_name].fillna(0,inplace=True)
    return train_set


def train_feats(orders,grouped_counts,grouped_means,dataset='training_data'):
    """
    Helper function for the add_features function below, and it extracts 
    features for the training set.
    Args:
        grouped_counts: The Grouped orders pandas dataframe, and values of the
                        different columns are counted.
        grouped_means: The Grouped orders pandas dataframe, and values of the
                       different columns are counted.
        dataset: This string indicates which dataset to consider. Is it the 
                 training_data dataset? Or the test_set_1, test_set_2? All 
                 these have different functions.
    Returns:
        final_train: The final training dataset, which is algorithm ready.
    """
    final_train = pd.DataFrame(columns=['week_day','second_price',
                                        'third_price','first_price',
                                        'second_gap','third_gap','first_gap',
                                        'second_order','third_order',
                                        'first_order'])
    week_day = grouped_means['week_day'][0]
    
    # Obtain the Gap
    gap = (grouped_counts.loc[:,'order_id'] - grouped_counts.loc[:,'driver_id'])
    grouped_counts.loc[:,'gap'] = gap
    gap = gap.sortlevel()

    # Extract the mean price for the whole dataset
    price = grouped_means.loc[:,'Price'].sortlevel()

    # Extract the number of orders per time slot for whole dataset
    orders = grouped_counts.loc[:,'order_id'].sortlevel()

    for d in range(1,67):
        means_idx = grouped_means.index.get_level_values('district_id') == d
        counts_idx = grouped_counts.index.get_level_values('district_id') == d

        district_means = grouped_means[means_idx]
        district_counts = grouped_counts[counts_idx]

        train_district = district_means.loc[:,['week_day']]

        # Extract the features and add them to the train set per district
        train_district = train_last_three(train_district, price, 'price')
        train_district = train_last_three(train_district, gap, 'gap')
        train_district = train_last_three(train_district, orders, 'orders')
        # import ipdb; ipdb.set_trace()

        if len(final_train) == 0:
            final_train = train_district
        else:
            final_train = pd.concat([final_train,train_district])
        
    final_train.sortlevel(inplace=True)
    final_train.loc[:,'gap'] = gap
    final_train.reset_index(inplace=True)
    return final_train


def add_last_three(test_set, test_previous, district, feature):
    """
    This function adds the last three time_slot transactions to the test set
    as three separate features.
    Args:
        test_set: The test set to which the last previous transactions are
                  added.
        test_previous: The dataframe containing the actual previous transaction
                       from which the last three are to be extracted
        feature: The feature to be added three times to the test set.

    Returns:
        test_set: The test set with the last three transactions already added
    """
    for pre in PREVIOUS:
        last_idx = test_set.index.get_level_values('time_slot') - PREVIOUS[pre]
        previous_idx = test_previous.index.get_level_values('time_slot')
        if len(test_previous) != 27:
            for idx in last_idx:
                if idx not in previous_idx:
                    # msg = ("Testset Preprocessing: {} idx for {} district data "
                    #       "is missing, now replacing it with the average of "
                    #       "zero").format(idx, district)
                    # print(msg)
                    test_previous.loc[(district,idx)] = 0
                    test_previous.sortlevel()
        previous_idx = test_previous.index.get_level_values('time_slot')
        mask = previous_idx.isin(last_idx)
        feat_data = test_previous.loc[mask].values
        test_set.loc[:,pre+'_'+feature] = feat_data
    # import ipdb; ipdb.set_trace()
    return test_set


def test_feats(orders, grouped_counts, grouped_means, dataset='test_set_1'):
    """
    Helper function for the function below, and it extracts features for the 
    testing set.
    Args:
        grouped_counts: The Grouped orders pandas dataframe, and values of the
                        different columns are counted.
        grouped_means: The Grouped orders pandas dataframe, and values of the
                       different columns are counted.
        dataset: This string indicates which dataset to consider. Is it the 
                 training_data dataset? Or the test_set_1, test_set_2? All 
                 these have different functions.
    Returns:
        final_data: The final data, which is algorithm ready.
    """
    # First, obtain the corresponding test set
    test_set = TEST_SETS[dataset]
    final_test = pd.DataFrame(columns=['Time','date','week_day','second_price',
                                       'third_price','first_price',
                                       'second_gap','third_gap','first_gap',
                                       'second_order','third_order',
                                       'first_order'])
    week_day = grouped_means['week_day'][0]
    previous_gap = (grouped_counts.loc[:,'order_id'] - 
                    grouped_counts.loc[:,'driver_id'])
    previous_orders = grouped_counts.loc[:,'order_id']
    previous_price = grouped_means['Price']
    # Check if all of the week days are the same.
    if not ((grouped_means['week_day'] == week_day).all()):
        import ipdb; ipdb.set_trace()
    test_day = test_set[test_set['week_day']==week_day]
    for i in range(1,67):
        gap_idx = previous_gap.index.get_level_values('district_id') == i
        orders_idx = previous_orders.index.get_level_values('district_id') == i
        price_idx = previous_price.index.get_level_values('district_id') == i 
        idx_test = test_day.index.get_level_values('district_id') == i
        
        gap_previous = previous_gap.loc[gap_idx]
        order_previous = previous_orders.loc[orders_idx]
        price_previous = previous_price.loc[price_idx]

        test_district = test_day.loc[idx_test]
        # Add the last time_slot prices, orders nums, and gaps to the test set
        test_district = add_last_three(test_district, gap_previous, i, 'gap')
        test_district = add_last_three(test_district, order_previous, i, 
                                       'order')
        test_district = add_last_three(test_district, price_previous, i, 
                                       'price')
        if len(final_test) == 0:
            final_test = test_district
        else:
            final_test = pd.concat([final_test,test_district])
    final_test.reset_index(inplace=True)
    # import ipdb; ipdb.set_trace()
    return final_test
        

def add_features(orders, dataset='training_data'):
    """
    Adds features and gap to the orders dataframe for a particular time slot.
    Args:
        orders: The order table, which contains the order data
        dataset: This string indicates which dataset to consider. Is it the 
                 training_data dataset? Or the test_set_1, test_set_2? All 
                 these have different functions.
    Returns:
        orders_with_feats: The order table, with features and gaps included.

    TODO: This function is quite repetitive, and can be improved. It needs some
          more elegance. Hopefully someone can make it more beautiful :)
    """

    # Group the order_data table by district_id and time_slot, and find means.
    grouped_counts = orders.groupby(['district_id', 'time_slot']).count()
    grouped_means = orders.groupby(['district_id', 'time_slot']).mean()
    
    grouped_counts = grouped_counts.sortlevel(level=0,axis=0)    
    grouped_means = grouped_means.sortlevel(level=0,axis=0)
    
    if dataset == 'training_data':
        final_data = train_feats(orders, grouped_counts, grouped_means)
    elif dataset[:4] == 'test':
        final_data = test_feats(orders, grouped_counts, grouped_means, 
                                dataset=dataset)
    return final_data


def process_orders(dataset_path, dataset, force):
    """
    A function to process the orders, and return a table of orders. This table 
    is the final table to be used and other features will be gradually added.
    Args:
        dataset_path: A String, which indicates the path to the folder which 
                    contains the different order tables for the different days
        dataset: A String which indicates which dataset this is. It can be 
                 training_data, test_set_1, or test_set_2

    Returns:
        clean_data: The table, which contains the order data, with the different
                    times and gaps included.
    """
    orders_path = os.path.join(dataset_path, 'order_data')
    order_files = os.listdir(orders_path)
    time_slots = create_slots()
    # Obtain the path for the hash to id table
    hash_table_path = os.path.join(dataset_path, 'cluster_map', 
                                   'cluster_map')
    hash_table = pd.read_table(hash_table_path, header=None, 
                               names=['Hash', 'ID'])
    if dataset == 'training_data':
        clean_data = pd.DataFrame(columns=['Time','date','week_day',
                                           'second_price',
                                           'third_price','first_price',
                                           'second_gap','third_gap',
                                           'first_gap',
                                           'second_order','third_order',
                                           'first_order', 'gap'])
    else:
        clean_data = pd.DataFrame(columns=['Time','date','week_day',
                                   'second_price',
                                   'third_price','first_price',
                                   'second_gap','third_gap',
                                   'first_gap',
                                   'second_order','third_order',
                                   'first_order'])
    for order_file in order_files:
        order_table_path = os.path.join(orders_path, order_file)
        print(order_file)
        # Ignore .DS_Store and Pickled Files
        if order_file.startswith('.') or '.pickle' in order_file:
            continue
        
        if os.path.exists(order_table_path+'.pickle') and not force:
            print('%s already present - Skipping extraction of %s' % (
                                    order_table_path+'.pickle', order_file))
            with open(order_table_path+'.pickle', 'rb') as f:
                orders = pkl.load(f)
        else:
            print('Now Extracting Data from %s' % (order_file,))
            orders = pd.read_table(order_table_path, header=None, 
                                   names=['order_id','driver_id',
                                          'passenger_id',
                                          'start_district_hash',
                                          'dest_district_hash',
                                          'Price', 'Time'])
            ### Here's Where Some of the Feature Extraction is Done.
            orders.loc[:,'time_slot'] = orders['Time'].apply(allocate_slots, 
                                                       args=(time_slots,))
            # Obtain the date string for a particular order file
            date_string = order_file[11:21]
            orders.loc[:,'week_day'] = pd.Series([get_week_day(date_string)]*
                                           len(orders))
            orders.loc[:,'district_id'] = orders['start_district_hash'].apply(
                                                                hash_to_id,
                                                        args=(hash_table,))
            with open(order_table_path+'.pickle', 'wb') as f:
                pkl.dump(orders, f)
            
        final_orders = add_features(orders, dataset)
        # import ipdb; ipdb.set_trace()
        if len(clean_data) == 0:
            clean_data = final_orders
        else:
            clean_data = pd.concat([clean_data,final_orders])
    import ipdb; ipdb.set_trace()
    return clean_data


if __name__ == '__main__':
    train_orders = load_data('public_data', 'training_data', force=False)
    # test_orders = load_data('public_data','test_set_1',force=False)
    # time_slots = create_slots()






