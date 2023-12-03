# FlappyBird
Harvard CS 184 Final Project, Fall 2023

Uses the [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) environment

To get started, run:  
* `conda create -n 184-project -c conda-forge -y python=3.11`
* `conda activate 184-project`
* `pip install -r requirements.txt`

## Behavioral Cloning (BC)
In this section, we have implemented the behavioral cloning (BC) algorithm for imitation learning. The code will record the user's actions while playing Flappy Bird and learn a policy based on those actions. To do this, first run
* `cd ./BC/`
* `python record.py`

This will open a pop-up window with Flappy Bird. Happy playing! You can modify the number of games to play with the `--n_games` argument to `record.py`.You can also record the actions of a pre-trained PPO model by specifying `--use_ppo_model=1`. Once this is done, you can train the model by running
* `python train.py`

Lastly, test the model by running
* `python test.py`

This will automatically run 10 tests, but you can modify the number of tests with the `--n_tests` argument to `test.py`.
