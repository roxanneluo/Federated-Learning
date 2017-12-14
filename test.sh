port=9000
#( CUDA_VISIBLE_DEVICES="" python ea_server.py $port 1 0.1 & );
for i in {1..50}
do
        ( CUDA_VISIBLE_DEVICES="" python ea_client.py $port $i & );
done
