import numpy as np
import keras
import random
import time
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace

import datasource
import threading



class LocalModel(object):
    # https://github.com/fchollet/keras/issues/2226
    def __init__(self, model_config):
        pass

    def gradient_step(self, batch_size, num_steps):
        pass



# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 50

    def __init__(self, server_host, server_port, datasource):
        ########## Socket IO messaging ##########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_init(*args):
            # we need:
            #   min data size
            #   model archetecture
            #   model initital param
            print('on init', args)

        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', on_init)

        self.sio.emit('wake_up')
        self.sio.wait(seconds=1)

        ########### lock ###########
        self.lock = threading.Lock()

        ####### dataset init #######
        self.datasource = datasource()

        # the local dataset can increase while training
        # [(Xi, Yi)]
        self.collected_data = []
        
        # weights for a random non-iid distribution
        # TODO: test extreme cases
        self.my_class_distr = np.array([np.random.random() for _ in range(self.datasource.classes.shape[0])])
        self.my_class_distr /= np.sum(self.my_class_distr)

        def simulate_data_gen(self):
            num_items = random.randint(10, FederatedClient.MAX_DATASET_SIZE_KEPT * 2)
            for _ in range(num_items):
                with self.lock:
                    # (X, Y)
                    self.collected_data += [self.datasource.sample_single_non_iid()]
                    # throw away older data if size > MAX_DATASET_SIZE_KEPT
                    self.collected_data = self.collected_data[-FederatedClient.MAX_DATASET_SIZE_KEPT:]
                    print(self.collected_data[-1][1])
                self.intermittently_sleep(p=.2, low=1, high=3)

        threading.Thread(target=simulate_data_gen, args=(self,)).start()

    
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