'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

class Client:
    # train/test_indices sample indices of train and test samples in this client
    def __init__(self, train_indices, test_indices):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
                self.prepare_dataset(train_indices, test_indices)
        self.model = self.build_model()

    def prepare_dataset(dataset_size, IID):
        pass

    # return dataset size, final weights
    def train(self, weights, epochs, batch_size):
        self.model.set_weights(weights)
        self.model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test))
        return self.x_train.shape[0], self.model.get_weights()

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score

class MnistClient(Client):
    img_rows, img_cols = 28, 28
    input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' \
            else (img_rows, img_cols, 1)
    num_classes = 10

    def prepare_dataset(self, train_indices, test_indices):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        img_rows, img_cols = MnistClient.img_rows, MnistClient.img_cols

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # get samples of this client
        x_train, y_train = x_train[train_indices, :,:,:], y_train[train_indices]
        x_test, y_test = x_test[test_indices], y_test[test_indices]


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, MnistClient.num_classes)
        y_test = keras.utils.to_categorical(y_test, MnistClient.num_classes)

        return (x_train, y_train), (x_test, y_test)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=MnistClient.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(MnistClient.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


if __name__ == "__main__":
    client = MnistClient(range(1000), range(1000))
    weights = client.train(client.model.get_weights(), 1, 128)
    client.evaluate()
