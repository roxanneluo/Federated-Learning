from fl_server import *
import sys
import threading

def print_request(head, req):
    print(head)
    [print(k, req[k]) for k in req if k != "weights"]

def sq_diff(w1, w2):
    s = 0
    sq_norm = lambda x: np.inner(x.reshape(-1), x.reshape(-1))
    for ww1, ww2 in zip(w1, w2):
        s += sq_norm(ww1-ww2)
    return s


class GlobalModel_MNIST_CNN_EASGD(GlobalModel_MNIST_CNN):
    def update_weights(self, client_weights, client_size_ratio, elasticity):
        #FIXME weight by total size
        new_weights = [gw + elasticity*client_size_ratio * (w-gw) for gw, w in
                zip(self.current_weights, client_weights)]
        self.current_weights = new_weights

class ClientMetadata:
    def __init__(self):
        self.meta = {}

    # data is a dict
    def set(self, ID, data):
        if ID not in self.meta:
            self.meta[ID] = {}
        self.meta[ID].update(data)

    def remove(self, ID):
        if ID in self.meta:
            del self.meta[ID]

    def sum(self, key):
        s = 0
        for ID in self.meta:
            s += self.meta[ID][key]
        return s

    def ratio(self, ID, key):
        val = self.meta[ID][key]
        return float(val) / self.sum(key)

    def get_all(self, keys):
        values = [[] for k in keys]
        for entry in self.meta.values():
            if not np.array([k in entry for k in keys]).all():
                continue
            for i, k in enumerate(keys):
                values[i].append(entry[k])
        return values


class ElasticAveragingServer(FLServer):
    def __init__(self, global_model, host, port, p, e):
        super(ElasticAveragingServer, self).__init__(global_model, host, port)
        # probability to synchronize. Note: here epoch_per_round ~ 1/p
        self.p = p
        self.e = e   # weight for elasiticity term
        self.model_lock = threading.Lock()

        self.client_metadata = ClientMetadata()


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
        def handle_disconnect():
            client_id = request.sid
            print(client_id, "disconnected")
            self.client_metadata.remove(client_id)
            print(self.client_metadata)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            emit('init', self.init_client_message())

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            client_id = request.sid
            self.client_metadata.set(client_id, data)
            print('client_ready', client_id, self.client_metadata)

        @self.socketio.on('client_request_weights')
        def handle_request_weights():
            print('client_request_weights')
            with self.model_lock:
                w = self.global_model.current_weights
            emit('server_send_weights', {'weights': obj_to_pickle_string(w)})

        @self.socketio.on('client_send_weights')
        def handle_client_send_weights(data):
            client_id = request.sid
            print_request('client %s _send_weights' % client_id, data)
            train_size_ratio = self.client_metadata.ratio(client_id,'train_size')
            w = pickle_string_to_obj(data["weights"])
            print('train_size_ratio', train_size_ratio)
            with self.model_lock:
                self.global_model.update_weights(w, train_size_ratio, self.e)
                gw = self.global_model.current_weights

            # update client_metadata
            result = data
            del result["weights"]

            for prefix in ['train', 'valid']:
                if '%s_loss' % prefix in result:
                    # compute real loss
                    #result["train_loss"] += self.e/2*sq_diff(w, gw)
                    self.client_metadata.set(client_id, result)
                    losses, accs, sizes = self.client_metadata.get_all(
                            [prefix+suf for suf in ['_loss' , '_accuracy', '_size']])
                    agg_loss, agg_acc = self.global_model.aggregate_loss_accuracy(losses, accs, sizes)
                    print(prefix + " results:", agg_loss, agg_acc)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.

    port = sys.argv[1]
    server = ElasticAveragingServer(GlobalModel_MNIST_CNN_EASGD, "127.0.0.1", int(port), 1, 0.1)
    print("listening on 127.0.0.1:" + str(port));
    server.start()
