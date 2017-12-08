import numpy as np
import keras
import random
import time
import threading
from fl_client import FederatedClient

class ElasticAveragingClient(FederatedClient):
    def __init__(self, server_host, server_port, datasource):
        # probability to synchronize. Note: here epoch_per_round ~ 1/p
        self.p = None
        self.e = None   # weight for elasiticity term
        FederatedClient.__init__(self, server_host, server_port, datasource)


    def on_init(self, *args):
        print('EA on_init')
        FederatedClient.on_init(self, *args)
        model_config = args[0]
        self.p = model_config["p"]
        self.e = model_config["e"]

        def synchronize():
            global_w = self.request_weights()
            local_w = self.local_model.get_weights()
            diff = [self.e * (w-gw) for w,gw in zip(local_w, global_w)]
            self.local_model.set_weights([w-d for w,d in zip(local_w, diff)])
            self.send_diff(diff)

        def train():
            while True:
                if random.random() < self.p:
                    synchronize()
                self.local_model.train_one_round()

        threading.Thread(target = train).start()

    def request_weights(self):
        #TODO
        pass

    def send_diff(self, diff):
        #TODO
        pass


if __name__ == "__main__":
    c = ElasticAveragingClient("127.0.0.1", 5000, datasource.Mnist)
