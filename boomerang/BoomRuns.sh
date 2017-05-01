#!/bin/bash
for i in {1..4}
do
   echo $i
   python boomerang.py -dt 0.01 -N 300001 -scheme FIXMAN --data-name ex-$i
done
