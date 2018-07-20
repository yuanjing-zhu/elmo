import pandas as pd
from collections import defaultdict
import itertools
import numpy as np
import torch as t


class DataIterator:
    def __init__(self,batch_size):
        self._batch_size = batch_size


    def get_batch(self, data_file_path):
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.
        """
        file_data = pd.read_json(data_file_path, lines=True, chunksize=self._batch_size)
        for batch in file_data:

            batch = self.pack_batch(batch)
            batch = self.pad_batch(batch)
            yield batch


    def print_feature_name(self, new_batch):
        """
        print all the keys and sub_keys of batch(after pack_batch)
        """
        for key in new_batch.keys():
            print(key)
            if type(new_batch[key]) == dict:
                print(new_batch[key].keys())


    def pack_batch(self, batch):
        new_batch = defaultdict(dict)
        for key in batch.iloc[0].keys():
            if type(batch.iloc[0][key]) == dict:
                for sub_key in batch.iloc[0][key].keys():
                    new_batch[key][sub_key] = [row[sub_key] for row in batch[key]]
            else:
                new_batch[key] = [getattr(row, key) for row in batch.itertuples(index=True, name='Pandas')]
        return new_batch


    def pad_value_list(self, value_list):
        value_list = list(itertools.zip_longest(*value_list, fillvalue=0))
        value_array = np.asarray(value_list).transpose()
        value_tensor = t.from_numpy(value_array)
        return value_tensor


    def pad_batch(self, new_batch):
        for key in new_batch.keys():
            for sub_key in new_batch[key]:
                new_batch[key][sub_key] = self.pad_value_list(new_batch[key][sub_key])
        return new_batch



#####################################################
batch = {}

new_batch = defaultdict(dict)
for key in batch.iloc[0].keys():
    if type(batch.iloc[0][key]) == dict:
        for sub_key in batch.iloc[0][key].keys():
            new_batch[key][sub_key] = [row[sub_key] for row in batch[key]]
    else:
        new_batch[key] = [getattr(row, key) for row in batch.itertuples(index=True, name='Pandas')]


for index, row in batch.iterrows():
    print(row['fact'])

dict_of_df = {k: pd.DataFrame(v) for k,v in batch.items()}
df = pd.concat(dict_of_df, axis=1)

vlist=[(1,2,3),(2,3,4,2,1),(34,2,4,4)]
list(itertools.zip_longest(*vlist, fillvalue=0))

