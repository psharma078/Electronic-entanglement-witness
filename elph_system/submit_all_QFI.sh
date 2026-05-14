#!/bin/bash

for i in {1..16}
do
    sbatch run_QFI.sh $i
done
