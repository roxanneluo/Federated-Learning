import numpy as np
import keras
import random
import time
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace

import datasource
import threading

class LocalModel(object):
    def __init__(self, model_config, data_collected):
        # model_config:
            # 'model': self.global_model.model.to_json(),
            # 'model_id'
            # 'min_train_size'
            # 'data_split': (0.6, 0.3, 0.1), # train, test, valid
            # 'epoch_per_round'
            # 'batch_size'
        self.model_config = model_config
        train_data, test_data, valid_data = data_collected
        self.x_train = np.array([tup[0] for tup in train_data])
        self.y_train = np.array([tup[1] for tup in train_data])
        self.x_test = np.array([tup[0] for tup in test_data])
        self.y_test = np.array([tup[1] for tup in test_data])
        self.x_valid = np.array([tup[0] for tup in valid_data])
        self.y_valid = np.array([tup[1] for tup in valid_data])

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return dataset size, final weights
    def train_one_round(self):
        self.model.fit(self.x_train, self.y_train,
                  epochs=self.model_config['epoch_per_round'],
                  batch_size=self.model_config['batch_size'],
                  verbose=1,
                  validation_data=(self.x_test, self.y_test))
        return self.x_train.shape[0], self.model.get_weights()

    def validate(self):
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        print('Validate loss:', score[0])
        print('Validate accuracy:', score[1])
        return score

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score


# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 1000

    def __init__(self, server_host, server_port, datasource):
        self.local_model = None
        self.datasource = datasource()

        ########## Socket IO messaging ##########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_init(*model_config):
            print('on init', model_config)
            # ([(Xi, Yi)], [], []) = train, test, valid
            fake_data = self.datasource.fake_non_iid_data(
                min_train=model_config['min_train_size'],
                max_train=FederatedClient.MAX_DATASET_SIZE_KEPT,
                data_split=model_config['data_split']
            )
            self.local_model = LocalModel(model_config, fake_data)


        def on_request_update(*req):
            # req:
            #     'model_id'
            #     'round_number'
            #     'current_weights'
            #     'weights_format'
            #     'run_validation'
            weights = # recover
            self.local_model.set_weights(weights)
            my_weights = self.local_model.train_one_round()
            if req['run_validation']:
                loss, accuracy = self.local_model.validate()
            self.sio.emit('client_update', {
                    # TODO
                })



        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', on_init)
        self.sio.on('request_update', on_request_update)

        self.sio.emit('wake_up')
        self.sio.wait(seconds=1)



        # TODO: later: simulate datagen for long-running train-serve service
        # i.e. the local dataset can increase while training

        # self.lock = threading.Lock()
        # def simulate_data_gen(self):
        #     num_items = random.randint(10, FederatedClient.MAX_DATASET_SIZE_KEPT * 2)
        #     for _ in range(num_items):
        #         with self.lock:
        #             # (X, Y)
        #             self.collected_data_train += [self.datasource.sample_single_non_iid()]
        #             # throw away older data if size > MAX_DATASET_SIZE_KEPT
        #             self.collected_data_train = self.collected_data_train[-FederatedClient.MAX_DATASET_SIZE_KEPT:]
        #             print(self.collected_data_train[-1][1])
        #         self.intermittently_sleep(p=.2, low=1, high=3)

        # threading.Thread(target=simulate_data_gen, args=(self,)).start()

    
    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))




# possible: use a client-level pubsub system for gradient update, no parameter server?
# e.g. Google cloud pubsub, Amazon SNS
# https://developers.google.com/nearby/connections/overview
# https://pypi.python.org/pypi/pyp2p

# class PeerToPeerClient(FederatedClient):
#     def __init__(self):
#         super(PushBasedClient, self).__init__()    



if __name__ == "__main__":
    c = FederatedClient("127.0.0.1", 5000, datasource.Mnist)