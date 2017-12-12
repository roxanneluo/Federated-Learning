import numpy as np
import keras
import random
from keras.datasets import mnist
from keras import backend as K

class DataSource(object):
    def __init__(self):
        raise NotImplementedError()
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()
    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()

class Mnist(DataSource):

    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 10
    
    TRAIN_VALID_DATA_RANGE = np.array([500,700])
    CLASSES = range(10)

    def __init__(self, train_range = (0.7 * TRAIN_VALID_DATA_RANGE).astype(int),
            valid_range = (0.3 * TRAIN_VALID_DATA_RANGE).astype(int),
            test_range = [80,120]):
        train_size = random.randint(train_range[0], train_range[1])
        test_size = random.randint(test_range[0], test_range[1])
        valid_size = random.randint(valid_range[0], valid_range[1])

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        total_train_size, total_test_size = x_train.shape[0], x_test.shape[0]

        total_valid_size = int(total_train_size * .3)
        total_train_size = int(total_train_size * .7)

        if Mnist.IID:
            train_sample_idx = np.random.choice(total_train_size, train_size,replace=True)
            valid_sample_idx = np.random.choice(range(total_train_size, total_train_size + total_valid_size), valid_size, replace=True)
            test_sample_idx = np.random.choice(total_test_size, test_size, replace=True)
        else:
            label_w = self.gen_dummy_non_iid_weights()
            # print('label_w', label_w)

            train_weights = self.gen_sample_weights(y_train, label_w)
            valid_weights = train_weights[total_train_size:] / np.sum(train_weights[total_train_size:])
            train_weights = train_weights[0:total_train_size] / np.sum(train_weights[0:total_train_size])
            test_weights = self.gen_sample_weights(y_test, label_w)

            train_sample_idx = np.random.choice(total_train_size, train_size,
                    replace=True, p=train_weights)
            valid_sample_idx = np.random.choice(range(total_train_size, total_train_size + total_valid_size), valid_size,
                    replace=True, p=valid_weights)
            test_sample_idx = np.random.choice(total_test_size, test_size,
                    replace=True, p=test_weights)

        self.x_train, self.y_train = self.post_process(
                x_train[train_sample_idx], y_train[train_sample_idx])
        self.x_valid, self.y_valid= self.post_process(
            x_train[valid_sample_idx], y_train[valid_sample_idx])
        self.x_test, self.y_test = self.post_process(
                x_test[test_sample_idx], y_test[test_sample_idx])

        print('train', self.x_train.shape, self.y_train.shape,
                'valid', self.x_valid.shape, 'test', self.x_test.shape)

    def gen_sample_weights(self,labels, label_w):
        size_per_class = np.array([np.sum(labels == i) for i in Mnist.CLASSES])
        label_w = np.divide(label_w, size_per_class)
        sample_w = np.zeros(labels.shape[0])
        sample_w[labels] = label_w
        sample_w /= np.sum(sample_w)
        return sample_w

    def gen_dummy_non_iid_weights(self):
        num_classes_this_client = random.randint(1, Mnist.MAX_NUM_CLASSES_PER_CLIENT + 1)
        classes_this_client = np.random.choice(Mnist.CLASSES, num_classes_this_client)
        w = np.random.rand(num_classes_this_client)

        weights = np.zeros(len(Mnist.CLASSES))
        weights[classes_this_client] = w
        return weights

    # assuming client server already agreed on data format
    def post_process(self, x, y):
        if K.image_data_format() == 'channels_first':
            sample_shape = (1, ) + x.shape[1:]
        else:
            sample_shape = x.shape[1:] + (1, )
        x = x.reshape((x.shape[0],) + sample_shape)

        y_vec = keras.utils.to_categorical(y, len(Mnist.CLASSES))
        return x / 255., y_vec

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

    # generate t, t, v dataset given distribution and split
    def fake_non_iid_data(self, min_train=100, max_train=1000, data_split=(.6,.3,.1)):
        return ((self.x_train, self.y_train),
                (self.x_test, self.y_test),
                (self.x_valid, self.y_valid))


if __name__ == "__main__":
    m = Mnist()
    # res = m.partitioned_by_rows(9)
    # print(res["test"][1].shape)
    for _ in range(10):
        print(m.gen_dummy_non_iid_weights())
