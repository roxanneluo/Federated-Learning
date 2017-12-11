port=9000
p=0.1
e=0.1
CUDA_VISIBLE_DEVICES="" python ea_server.py $port $p $e 20 "results/p$p_e$e.txt"
