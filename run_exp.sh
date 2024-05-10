#!/bin/bash

# Define the list of seeds
seeds=(100 200 300 400 500)

# Define the list of n-episodes
n_episodes=(80000 100000 150000 200000)

# Loop through each seed
for seed in "${seeds[@]}"; do
    # Loop through each n-episode
    for n in "${n_episodes[@]}"; do
        # Run the python script with the specified arguments
        python main.py -s "$seed" -n "$n"
    done
done

