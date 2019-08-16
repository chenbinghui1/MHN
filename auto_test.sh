#!/usr/bin/env sh
for j in $(seq 7)
do
i=`expr \( $j + 0 \) \* 10 - 1`
echo model,$i
python3 test_smallPCB.py --gpu_ids 0 --name pcb_mhn6 --test_dir datasets/Duke/datasets/pytorch/ --batchsize 13 --which_epoch $i --parts 6 --mhn
python3 evaluate_gpu.py --gpu_ids 0
done
