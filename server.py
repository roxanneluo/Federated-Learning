import threading
class Server:
    def __init__(self):
        self.weights = None # what to fill in here
        self.num_samples = 0
        self.lock = threading.Lock()

    def update_weights(self, num_samples, weights):
        with self.lock:
            self.num_samples += num_samples
            for sw, w in zip(self.weights, weights):
                sw += num_samples * w

    def get_weights(self):
        with self.lock:
            if self.num_samples == 0:
                return self.weights

            for w in self.weights:
                w /= self.num_samples
            self.num_samples = 0
            return self.weights
