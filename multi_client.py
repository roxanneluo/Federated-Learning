from fl_client import FederatedClient
import datasource
import multiprocessing
import threading
import random

def start_client(seed):
    print("start client")
    random.seed(seed)
    c = FederatedClient("127.0.0.1", 5000, datasource.Mnist)


if __name__ == '__main__':
    jobs = []
    for i in range(50):
        # threading.Thread(target=start_client).start()

        p = multiprocessing.Process(target=start_client, args=(i,))
        jobs.append(p)
        p.start()
    # TODO: randomly kill
