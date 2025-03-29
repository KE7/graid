#!/bin/bash
# This script iterates over combinations of dataset, model, and conf values.
# It assigns each job to a GPU in parallel (one per GPU) and saves the output
# to a file named by the combination of parameters.

datasets=("bdd") # "nuimage" "waymo"
models=("DINO" "Co_DETR" "yolo_v10x" "yolo_11x" "rtdetr" "retinanet_R_101_FPN_3x" "faster_rcnn_R_50_FPN_3x")
confs=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
gpus=(0 1 2 3 4 5 6 7)

max_jobs=8
job_count=0

# Iterate over all parameter combinations
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for conf in "${confs[@]}"; do
            gpu=${gpus[$(( job_count % max_jobs ))]}
            output_file="outputs/output_${dataset}_${model}_${conf}_gpu${gpu}.txt"
            echo "Launching job: dataset=${dataset}, model=${model}, conf=${conf}, gpu=${gpu}"
            mkdir -p outputs
            python coco_stream.py --dataset "$dataset" --model "$model" --conf "$conf" --device-id "$gpu" > "$output_file" 2>&1 &
            job_count=$((job_count+1))
            # Once we've launched a batch equal to the number of GPUs, wait for all to finish
            if (( job_count % max_jobs == 0 )); then
                wait
            fi
        done
    done
done

# Wait for any remaining jobs to finish before exiting
wait
