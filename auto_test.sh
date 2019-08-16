#!/usr/bin/env sh
for j in $(seq 7)
do
i=`expr \( $j + 0 \) \* 10 - 1`
echo model,$i
gpu_ids=0
python3 test_ide.py --gpu_ids $gpu_ids --name ide --test_dir datasets/Market/datasets/pytorch/ --batchsize 32 --which_epoch $i
python3 evaluate_gpu.py --gpu_ids $gpu_ids
done
