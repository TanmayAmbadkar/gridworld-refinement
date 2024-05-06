#!/bin/bash

# Define the list of seeds
seeds=(0 10 20 30 40)

# Define the list of n-episodes
n_episodes=(5000 10000 50000 100000)

# Loop through each seed
for seed in "${seeds[@]}"; do
    # Loop through each n-episode
    for n in "${n_episodes[@]}"; do
        # Run the python script with the specified arguments
        python main.py -s "$seed" -n "$n"
    done
done
