#!/usr/bin/env python
# coding: utf-8

# Central questions: can classifiers be trained using neuroimaging meta-analysis data (i.e., coordinates, modeled activation maps, etc.). Goals: [1] train classifier using coordinates/modeled activation maps, [2]...

# Import libraries and packages.
from nimare.meta.cbma.kernel import ALEKernel
from nimare.io import convert_sleuth_to_dataset
import pandas as pd
import numpy as np
import glob
import datetime
today = datetime.date.today()

# Select directories and files structure.
input_prefix = str(today)
output_prefix = str(today)
in_dir = 'constructs'
out_dir = 'out'
paths = glob.glob("constructs/*.txt")
print('constructs to model = {0}'.format(len(paths)))

# Convert coordinates to nimare dataset. DOES NOT WORK!
datas = {}
for path in paths:
    print(path)
    datas[path[len(in_dir) + 1:-4]] = convert_sleuth_to_dataset(path)

datas.keys()  # Confirm construct keys.

# Make modeled activation (MA) maps.
print('MA maps making...\t\t@{0}'.format(str(datetime.datetime.now())))
ma_maps_arrs = {}
for data in datas.keys():
    kern = ALEKernel()
    ma_maps = kern.transform(datas[data])  # Compute MA maps (len = ???)
    ma_maps_arrs[data] = []
    for i in np.arange(0, len(ma_maps)):
        ma_maps_arrs[data].append(np.ravel(ma_maps[i].get_data(), order='C'))
    labels = pd.DataFrame(index=datas[data].ids)
print('MA maps done!\t\t\t@{0}'.format(str(datetime.datetime.now())))
# print('MA maps = {0}'.format(len(ma_maps_arr)))
# print('MA maps shape = {0}'.format(ma_maps_array.shape))

keys = list(datas.keys())
dataframes = {}
key = {}
for i in np.arange(0, len(keys)):
    key[keys[i]] = i
    arr = np.asarray(ma_maps_arrs[keys[i]])
    dataframes[i] = pd.DataFrame(arr)
    dataframes[i]['y'] = i

all_data = pd.concat([dataframes[0], dataframes[1]], ignore_index=True)

for i in np.arange(2, len(dataframes.keys())):
    all_data = pd.concat([all_data, dataframes[i]], ignore_index=True)

all_data.to_csv('all_data_df.csv')
