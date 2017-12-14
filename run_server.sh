port=9000
p=0.1
e=0.8
CUDA_VISIBLE_DEVICES="" python ea_server.py $port $p $e 20 "results/p"$p"_e$e.txt"
