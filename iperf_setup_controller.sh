#!/bin/sh

# First argument is taken as time in seconds
TIME=${1:-600}

sudo iperf3 -c 192.168.0.3 -p 5001 -t "$TIME" -S 0 &>/dev/null &
sudo iperf3 -c 192.168.0.3 -p 5002 -t "$TIME" -S 32 &>/dev/null &
sudo iperf3 -c 192.168.0.3 -p 5003 -t "$TIME" -S 56 &>/dev/null &
sudo iperf3 -c 192.168.0.3 -p 5004 -t "$TIME" -S 88 &>/dev/null &
sudo iperf3 -c 192.168.0.3 -p 5005 -t "$TIME" -S 112 &>/dev/null &
sudo iperf3 -c 192.168.0.3 -p 5006 -t "$TIME" -S 152 &>/dev/null &
sudo iperf3 -c 192.168.0.3 -p 5007 -t "$TIME" -S 184 &>/dev/null &
sudo iperf3 -c 192.168.0.3 -p 5008 -t "$TIME" -S 224 &>/dev/null & 