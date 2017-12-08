import pickle
import keras
import uuid
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import msgpack
import random
import codecs
import msgpack_numpy
# https://github.com/lebedov/msgpack-numpy

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/
       


def obj_to_pickle_string(obj):
    return pickle.dumps(obj)
    # codecs.encode(pickle.dumps(obj), "base64").decode()
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(s)
    # pickle.loads(codecs.decode(s, "base64"))


class GlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        # for convergence check
        self.prev_train_loss = None
    
    def build_model(self):
        raise NotImplementedError()

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)

        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        self.current_weights = new_weights        

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuraries
        

class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
        # ~5MB worth of parameters
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

        
######## Flask server with Socket IO ########

# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):
    
    MIN_NUM_WORKERS = 1
    MAX_NUM_ROUNDS = 10
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1
    ROUNDS_BETWEEN_VALIDATIONS = 10

    def __init__(self, global_model, host, port):
        self.global_model = global_model()

        self.ready_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        self.model_id = str(uuid.uuid4())

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        #####

        self.register_handles()


    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            emit('init', {
                    'model_json': self.global_model.model.to_json(),
                    'model_id': self.model_id,
                    'min_train_size': 100,
                    'data_split': (0.6, 0.3, 0.1), # train, test, valid
                    'epoch_per_round': 1,
                    'batch_size': 10
                })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training", request.sid, data)
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:
                self.train_next_round()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            print("handle client_update", request.sid)
            for x in data:
                if x != 'weights':
                    print(x, data[x])
            # data:
            #   weights
            #   train_size
            #   valid_size
            #   train_loss
            #   train_accuracy
            #   valid_loss?
            #   valid_accuracy?

            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                
                # tolerate 30% unresponsive clients
                if self.current_round_update_received > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )
                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )
                    print("aggr_train_loss", aggr_train_loss)
                    print("aggr_train_accuracy", aggr_train_accuracy)

                    if 'valid_loss' in self.current_round_client_updates[0]:
                        aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_loss_accuracy(
                            [x['valid_loss'] for x in self.current_round_client_updates],
                            [x['valid_accuracy'] for x in self.current_round_client_updates],
                            [x['valid_size'] for x in self.current_round_client_updates],
                        )
                        print("aggr_valid_loss", aggr_valid_loss)
                        print("aggr_valid_accuracy", aggr_valid_accuracy)

                    if self.global_model.prev_train_loss is not None and \
                            (self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss < .01:
                        # converges
                        print("converges!")
                        self.stop()
                    else:
                        self.global_model.prev_train_loss = aggr_train_loss
                        self.train_next_round()

    
    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        print("### Round ", self.current_round, "###")
        client_sids_selected = random.sample(list(self.ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        print("request updates from", client_sids_selected)

        # by default each client cnn is in its own "room"
        for rid in client_sids_selected:
            emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),

                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=rid)

    def stop_training(self):
        emit('stop', {
                'model_id': self.model_id,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle'
            })

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)



if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    
    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000");
    server.start()
