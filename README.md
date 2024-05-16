# FlappyBird
Harvard CS 184 Final Project, Fall 2023

Uses the [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) environment

<img width="400" alt="flappy-bird-res" src="https://github.com/SantiagoGiner/FlappyBird/assets/70669841/7ffabaf3-5286-4ecf-b52b-c02eac0feb3e">

Check out [our paper](https://drive.google.com/file/d/1HYpYvKdjh9Y-aS0YRDdeYHeYGywLcYL_/view?usp=sharing) summarizing our findinds
<img width="790" alt="Screenshot 2024-05-16 at 3 40 26 PM" src="https://github.com/SantiagoGiner/FlappyBird/assets/70669841/b691011f-cd06-416e-946a-07a24a3fb59c">

## Usage
To get started, run:  
* `conda create -n 184-project -c conda-forge -y python=3.11`
* `conda activate 184-project`
* `pip install -r requirements.txt`

## Behavioral Cloning (BC)
In this section, we have implemented the behavioral cloning (BC) algorithm for imitation learning. The code will record the user's actions while playing Flappy Bird and learn a policy based on those actions. To do this, first run
* `cd ./BC/`
* `python record.py`

This will open a pop-up window with Flappy Bird. Happy playing! You can modify the number of games to play with the `--n_games` argument to `record.py`.You can also record the actions of a pre-trained PPO model by specifying `--use_ppo_model`. Once this is done, you can train the model by running
* `python train.py`

Lastly, test the model by running
* `python test.py`

This will automatically run 10 tests, but you can modify the number of tests with the `--n_tests` argument to `test.py`.
