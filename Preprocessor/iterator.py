

class DataIterator():
    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        self._batch_size = batch_size


    def get_batch(self, data_file_path: str = None):
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.
        """
        with open(data_file_path,'r') as f:
            batch = f.read(self._batch_size)
            yield batch