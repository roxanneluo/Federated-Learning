import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import msgpack
import msgpack_numpy
# https://github.com/lebedov/msgpack-numpy

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/
       

class GlobalModel_MNIST_CNN(object):
    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()

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
    
    MIN_NUM_WORKERS = 2
    MAX_NUM_ROUNDS = 10
    NUM_CLIENTS_CONTACTED_PER_ROUNT = 1

    def __init__(self, global_model, host, port):
        self.global_model = global_model()

        self.training_started = False
        self.connected_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")
            self.connected_client_sids.add(request.sid)
            if len(self.connected_client_sids) >= FLServer.MIN_NUM_WORKERS and not self.training_started:
                self.train_global_model()

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")
            self.connected_client_sids.add(request.sid)
            if len(self.connected_client_sids) >= FLServer.MIN_NUM_WORKERS and not self.training_started:
                self.train_global_model()

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            self.connected_client_sids.remove(request.sid)

        @self.socketio.on('wake_up')
        def handle_wake_up():
            print("handle wake_up")
            print("sid ", request.sid)
            # for w in global_model.model.get_weights():
            #     print(w.shape)

            emit('init', {
                    'model': self.global_model.model.to_json(),
                    'min_dataset_size': 100,
                    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO
                    # 'initial_weights': str(pickle.dumps(global_model.current_weights)),
                    # 'initial_weights': str(msgpack.packb(global_model.current_weights, default=msgpack_numpy.encode)),
                    # 'weights_format': 'pickle'
                })

    
    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_global_model(self):
        for r in range(FLServer.MAX_NUM_ROUNDS):
            print("### Round ", r, "###")
            client_sids_selected = random.sample(self.connected_client_sids, FLServer.NUM_CLIENTS_CONTACTED_PER_ROUNT)

            # !!!!!!!!!!!!!
            # TODO: lifecycle here
            # emit "request_gradient", with round#
            # handle "gradient_update", take average

            # every k rounds, kickoff federated validation; break if converge


    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    
    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000");
    server.start()
