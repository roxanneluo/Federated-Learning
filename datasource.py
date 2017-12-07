import numpy as np
import keras

from keras.datasets import mnist


class DataSource(object):
    def __init__(self):
        raise NotImplementedError()
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()
    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()


class Mnist(DataSource):
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x = np.concatenate([x_train, x_test]).astype('float')
        self.y = np.concatenate([y_train, y_test])
        n = self.x.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        self.x = self.x[idx]  # n * 28 * 28
        self.y = self.y[idx]  # n * 1
        self.classes = np.unique(self.y)
        
    # assuming client server already agreed on data format
    def post_process(self, xi, yi):
        y_vec = keras.utils.to_categorical(yi, self.classes.shape[0])
        return xi / 255., y_vec

    # split evenly into exact num_workers chunks, with test_reserve globally
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        n_test = int(self.x.shape[0] * test_reserve)
        n_train = self.x.shape[0] - n_test
        nums = [n_train // num_workers] * num_workers
        nums[-1] += n_train % num_workers
        idxs = np.array([np.random.choice(np.arange(n_train), num, replace=False) for num in nums])
        return {
            # (size_partition * 28 * 28, size_partition * 1) * num_partitions
            "train": [post_process(self.x[idx], self.y[idx]) for idx in idxs],
            # (n_test * 28 * 28, n_test * 1)
            "test": post_process(self.x[np.arange(n_train, n_train + n_test)], self.y[np.arange(n_train, n_train + n_test)])
        }

    # Generate one sample from all available data, *with replacement*.
    # This is to simulate date generation on a client.
    # weight: [probablity of classes]
    # returns: 28 * 28, 1
    def sample_single_non_iid(self, weight=None):
        # first pick class, then pick a datapoint at random
        chosen_class = np.random.choice(self.classes, p=weight)
        candidates_idx = np.array([i for i in range(self.y.shape[0]) if self.y[i] == chosen_class])
        idx = np.random.choice(candidates_idx)
        return self.post_process(self.x[idx], self.y[idx])

    
    # generate t, t, v dataset given distribution and split
    def fake_non_iid_data(self, min_train=100, max_train=1000, data_split=(.6,.3,.1)):
        my_class_distr = np.array([np.random.random() for _ in range(self.classes.shape[0])])
        my_class_distr /= np.sum(self.my_class_distr)
        
        train_size = random.randint(min_train, max_train)
        test_size = train_size / data_split[0] * data_split[1]
        valid_size = train_size / data_split[0] * data_split[1]

        train_set = [self.sample_single_non_iid(my_class_distr) for _ in train_size]
        test_set = [self.sample_single_non_iid(my_class_distr) for _ in test_size]
        valid_set = [self.sample_single_non_iid(my_class_distr) for _ in valid_size]
        return (train_set, test_set, valid_set)


if __name__ == "__main__":
    m = Mnist()
    # res = m.partitioned_by_rows(9)
    # print(res["test"][1].shape)
    print(m.sample_single_non_iid()[0].shape)
