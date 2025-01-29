#!/bin/bash

echo "Starting server"
python 5_clients_server.py &
sleep 10  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 5); do
    echo "Starting client $i"
    python client.py --id $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait