import pandas as pd 
# load my result
my_result = pd.read_table("ralph.csv", sep=',', names=['depart', 'time', 'gap'])
# load golden result
golden_result = pd.read_table("test_result_1", sep=',', names=['depart', 'time', 'gap'])

result = pd.merge(my_result, golden_result, on=['depart', 'time'], how='right')
#result.fillna(0, inplace = True)
result['gap'] = (result['gap_y'] - result['gap_x']).abs()
result = result[['depart', 'time', 'gap']]

print result.groupby('depart').mean().mean()

import ipdb; ipdb.set_trace()