#!/bin/bash
for i in {1..4}
do
   echo $i
   python multi_bodies_gmres_PC.py -dt 0.01 -N 300001 --data-name ETA1e-4-delta4-$i
done
