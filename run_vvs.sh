#!/bin/bash

prompt=MSCOCO2017Val

# target models
llamagen2_model=/home/to/your/path/LlamaGen-T2I-2
# draft models
llamagen2_drafter=/home/htdong/storage_net/sModels/llamagen2_drafter

skip_policies=uniform # choices: ['uniform', 'dynamic']
select_token_policies=random
reuse_feature_policies=reuse_prev

# interval list
interval_list=(5)
# choose to skip finished lab
skip_finished="1"

# Iterate over all combinations of policies
echo "Testing with skip_policy=$skip_policy,
select_token_policy=$select_token_policy, 
reuse_feature_policy=$reuse_feature_policy"

# Set intervals based on skip_policy
if [ "$skip_policy" != "uniform" ]; then
    intervals=(1024)
else
    intervals=("${interval_list[@]}")
fi

device=0

# Iterate over intervals
for interval in "${intervals[@]}"; do
    output_dir="output/${select_token_policy}_${reuse_feature_policy}_${skip_policy}_i${interval}_${prompt}"
    if [ "$skip_finished" = "1" ] && [ -d "$output_dir" ] && [ "$(find "$output_dir" -maxdepth 1 -type f -name '*.json' | wc -l)" -gt 0 ]; then
        echo "Experiment already run for $output_dir, skipping... ⚠️"
        echo "------------------------------------------------------"
        continue
    fi
    echo "Testing with interval=$interval"

    # execution of main program
    CUDA_VISIBLE_DEVICES=$device python main.py gen_images \
    --model llamagen2 \
    --model_type eagle \
    --model_path "${llamagen2_model}" \
    --output_dir $output_dir \
    --prompt $prompt \
    --drafter_path "${llamagen2_drafter}" \
    --target_size 1024 \
    --top_k 1000 \
    --top_p 1.0 \
    --temperature 1.0 \
    --slice 0-1 \
    --num_images 5000 \
    --cfg 7.5 \
    --lantern \
    --lantern_k 1000 \
    --lantern_delta 0.2 \
    --random_seed 42 \
    --skip \
    --skip_interval $interval \
    --skip_policy $skip_policy \
    --select_token $select_token_policy \
    --reuse_feature $reuse_feature_policy \

    echo "Finished testing with interval=$interval ✅"
    echo "------------------------------------------"
done