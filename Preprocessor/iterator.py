import json
import pandas as pd

class DataIterator():
    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        self._batch_size = batch_size


    def get_batch(self, data_file_path):
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.
        """
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = pd.DataFrame(json.loads(line) for line in f)
        batch_count = 0
        end = 0
        while end + self._batch_size < len(data):
            start = batch_count * self._batch_size
            end = start + self._batch_size
            batch_count += 1
            batch = data.iloc[start:end]
            batch = self.pad_batch(batch)
            yield batch

    def print_feature_name(self,batch):
        print(batch.iloc[0].keys())

    def pack_batch(self, batch):
        new_batch = {}
        for key in batch.iloc[0].keys():
            new_batch[key] = batch[key]
            if batch[key].keys():
                for sub_key in batch[key][0]:
                    new_batch[key][sub_key] = batch[key]
        return new_batch

    def pad_batch(self, batch):

        return batch










with open('data/data_train.json', 'r', encoding='utf-8') as f:
    data = pd.DataFrame(json.loads(line) for line in f)


