from fl_client import FederatedClient
import datasource
import multiprocessing
import threading

def start_client():
    print("start client")
    c = FederatedClient("127.0.0.1", 5000, datasource.Mnist)


if __name__ == '__main__':
    jobs = []
    for i in range(5):
        # threading.Thread(target=start_client).start()

        p = multiprocessing.Process(target=start_client)
        jobs.append(p)
        p.start()
    # TODO: randomly kill