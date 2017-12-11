import random
import sys
import threading
from fl_client import FederatedClient
from fl_server import obj_to_pickle_string, pickle_string_to_obj, FLServer
from ea_server import print_request
import datasource


class ElasticAveragingClient(FederatedClient):
    def __init__(self, server_host, server_port, datasource):
        # probability to synchronize. Note: here epoch_per_round ~ 1/p
        self.p = None
        self.e = None   # weight for elasiticity term
        self.model_lock = threading.Lock()
        self.result = {} # train/validation loss and accuracy
        self.alive = True

        super(ElasticAveragingClient, self).__init__(server_host, server_port, datasource)

    # register socket handles
    def register_handles(self):
        super(ElasticAveragingClient, self).register_handles()

        def on_server_send_weights(*args):
            req = args[0]
            print_request('on_server_send_weights', req)

            global_w = pickle_string_to_obj(req['weights'])
            with self.model_lock:
                local_w = self.local_model.get_weights()
                diff = [self.e * (w-gw) for w,gw in zip(local_w, global_w)]
                self.local_model.set_weights([w-d for w,d in zip(local_w, diff)])
                cur_result = self.result

            self.send_weights(local_w, cur_result)

        ## register handle
        self.sio.on('server_send_weights', on_server_send_weights)


    def on_init(self, *args):
        print_request('EA on_init', args[0])
        super(ElasticAveragingClient, self).on_init(*args)
        model_config = args[0]
        self.p = model_config["p"]
        self.e = model_config["e"]
        with self.model_lock:
            self.local_model.set_weights(pickle_string_to_obj(model_config["weights"]))

        def train():
            iteration = 0
            while self.alive:
                iteration += 1
                print('iter', iteration)
                if random.random() < self.p:
                    self.request_weights()
                with self.model_lock:
                    _, train_loss, train_accuracy = self.local_model.train_one_round()
                    self.result["train_loss"] = train_loss
                    self.result["train_accuracy"] = train_accuracy

                """
                if iteration % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0:
                    with self.model_lock:
                        valid_loss, valid_accuracy = self.local_model.validate()
                        self.result["valid_loss"] = valid_loss
                        self.result["valid_accuracy"] = valid_accuracy
                """

        threading.Thread(target = train).start()

    def request_weights(self):
        self.sio.emit('client_request_weights')

    def send_weights(self, weights, result):
        resp = {'weights': obj_to_pickle_string(weights),}
        resp.update(result)
        self.sio.emit('client_send_weights', resp)


if __name__ == "__main__":
    port = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else random.seed()
    random.seed(seed)
    c = ElasticAveragingClient("127.0.0.1", int(port), datasource.Mnist)
