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
    data = process_orders(dataset_path, force)
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
            time_slot = time_slots.index(slot)
            assign_slot += 1
    if assign_slot != 1: # Check if the slot was assigned twice for an order_id
        import ipdb; ipdb.set_trace()
    return time_slot


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
    week_day = date_datetime.weekday()
    return week_day


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


def add_gap(orders):
    """
    Adds gap to the orders dataframe for a particular time slot. This will be 
    the target for that time slot. The district and other variables will all be
    the same.
    Args:
        orders: The order table, which contains the order data
    Returns:
        orders_with_gap: The order table, with gaps included.

    TODO: This function is quite repetitive, and can be improved. It needs some
          more elegance. Hopefully someone can make it more beautiful :)
    """
    # First, obtain the orders where the driver_id is null
    null_orders = orders[orders['driver_id'].isnull()]
    null_counts = null_orders.groupby(['district_id', 'time_slot']).count()
    null_averages = null_orders.groupby(['district_id', 'time_slot']).mean()
    # gap_dict = dict(grouped.iloc[:,0])
    null_counts.reset_index(inplace=True)
    null_averages.reset_index(inplace=True)
    null_counts['gap'] = null_counts['start_district_hash']
    null_counts['week_day'] = null_averages['week_day']
    null_counts['mean_price'] = null_averages['Price']
    # null_counts['mean_price'] = 
    del null_counts['start_district_hash']
    del null_counts['dest_district_hash']
    del null_counts['Price']
    del null_counts['Time']
    del null_counts['order_id']
    del null_counts['driver_id']
    del null_counts['passenger_id']
    # del null_counts['week_day']
    # import ipdb; ipdb.set_trace()
    return null_counts


def process_orders(dataset_path, force):
    """
    A function to process the orders, and return a table of orders. This table 
    is the final table to be used and other features will be gradually added.
    Args:
        order_path: A String, which indicates the path to the folder which 
                    contains the different order tables for the different days

    Returns:
        clean_data: The table, which contains the order data, with the different
                    times and gaps included.
    """
    orders_path = os.path.join(dataset_path, 'order_data')
    order_files = os.listdir(orders_path)
    time_slots = create_slots()
    clean_data = pd.DataFrame(columns=['time_slot', 'week_day', 'district_id',
                                       'mean_price', 'gap'])
    # Obtain the path for the hash to id table
    hash_table_path = os.path.join(dataset_path, 'cluster_map', 
                                   'cluster_map')
    hash_table = pd.read_table(hash_table_path, header=None, 
                               names=['Hash', 'ID'])
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
            orders['time_slot'] = orders['Time'].apply(allocate_slots, 
                                                       args=(time_slots,))
            # Obtain the date string for a particular order file
            date_string = order_file[11:21]
            orders['week_day'] = pd.Series([get_week_day(date_string)]*
                                           len(orders))

            orders['district_id'] = orders['start_district_hash'].apply(
                                                                hash_to_id,
                                                        args=(hash_table,))
            orders = add_gap(orders)
            # orders['week_day'] = 
            with open(order_table_path+'.pickle', 'wb') as f:
                pkl.dump(orders, f)

        clean_data = clean_data.append(orders, ignore_index=True)
        # import ipdb; ipdb.set_trace()
    return clean_data


if __name__ == '__main__':
    load_data('public_data', 'training_data', force=False)
    # time_slots = create_slots()






