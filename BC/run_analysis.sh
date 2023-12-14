#!/bin/bash

# Try out different num_dataset_samples
python record.py --use_ppo_model --n_games 20 --data_name ppo
for n in $(seq 100 11000 1000); do
    echo "Recording for ${n}...";
    python train.py --data_name ppo --num_dataset_samples "$n";
    echo "Testing for ${n} ...";
    python test.py --n_tests 10 --hide_game;
done

# # Try out different epoch numbers
# learning_rates=(0.15)
# python record.py --use_ppo_model --n_games 20 --data_name ppo;
# for n in "${learning_rates[@]}"; do
#     # Call the Python script with the current value of num_samples
#     echo "Recording using ${n} lr...";
#     python train.py --data_name ppo --lr "$n" --plot_dir ./plots/ --plot_name "BC_$n";
#     echo "Testing using ${n} lr...";
#     python test.py --hide_game;
# done
