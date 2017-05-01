#!/bin/bash
N=8
for n in `seq 1 $N`;
do
rm inputfile.disp."$n"
echo "I just removed file " $n " for you"
done
