from fl_server import FederatedServer

class GlobalModel_MNIST_CNN_EASGD(GlobalModel_MNIST_CNN):
    def update_weights(self, client_weights, client_sizes, elasticity):
        #FIXME weight by total size
        new_weights = [gw + elasticity*(gw-w) for gw, w in
                zip(self.current_weights, client_weights)]
        self.current_weights = new_weights        

def ElasticAveragingServer(FLServer):
    def __init__(self, global_model, host, port, p, e):
        super(ElasticAveragingServer, self).__init__(global_model, host, port)
        # probability to synchronize. Note: here epoch_per_round ~ 1/p
        self.p = p
        self.e = e   # weight for elasiticity term
        self.model_lock = threading.Lock()

    def init_client_message(self):
        msg = super(ElasticAveragingServer, self).init_client_message()
        msg["epoch_per_round"] = 1
        msg["e"] = self.e
        msg["p"] = self.p
        return msg 

    def register_handles(self):
        print('EA register handles')
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
            emit('init', self.init_client_message())

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            pass

        @self.socketio.on('client_request_weights')
        def handle_request_weights():
            with self.model_lock:
                w = self.global_model.current_weights
            emit('server_send_weights', {'weights': w})

        @self.socketio.on('client_send_weights')
        def handle_client_send_weights(data):
            with self.model_lock:
                self.global_model.update_weights(data['weights'],
                        data['train_size'], self.e)

if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    
    server = ElasticAveragingServer(GlobalModel_MNIST_CNN_EASGD, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000");
    server.start()
