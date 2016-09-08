"""
This file is to preprocess the test set for the didi competition
"""

import pandas as pd
import numpy as np
# from didi_input import load_data
import didi_input
import os

def get_date(time):
    """
    A helper function for the function load_test below to get the date
    """
    date = time[:10]
    return date

# def get_

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
    test_result = pd.read_table(test_set_path, header=None, delimiter=',',
                                names=['district_id','Time','gap'])

    test_times = pd.DataFrame(test_result.Time.str.rsplit('-',1).tolist(),
                              columns=['date', 'time_slot'])
    test_result = pd.concat([test_result, test_times], axis=1)
    del test_result['Time']
    test_result['week_day'] = test_result['date'].apply(
                                didi_input.get_week_day)
    test_result['time_slot'] = pd.to_numeric(test_result['time_slot'])
    # del test_result['date']
    # test_result['week_day'] = test_result['week_day'].apply(
    #                             didi_input.get_week_day)
    # test_set = pd.
    return test_result

    # import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    # test_result = load_test('public_data', force=False)
    test_orders = didi_input.load_data('public_data','test_set_1/read_me_1.txt',
                                       force=False)

