from neural_agent import NeuralAgent, CommandScorer
from random_agent import RandomAgent
from lstm_neural_agent import NeuralAgentLSTM, CommandScorerLSTM
from transformers_neural_agent import TransformerNeuralAgent, TransformerScorer
import os
from glob import glob
from typing import Mapping, Any
import numpy as np
import textworld.gym
import torch
from time import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def play(agent, path, max_step=100, num_episodes=10, verbose=True):
    """
    Based on code from the TextWorld repository: https://github.com/microsoft/TextWorld
    """
    torch.manual_seed(20211021)  # For reproducibility when using action sampling.

    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.

    gamefiles = [path]
    if os.path.isdir(path):
        gamefiles = glob(os.path.join(path, "*.z8"))

    env_id = textworld.gym.register_games(gamefiles,
                                          request_infos=infos_to_request,
                                          max_episode_steps=max_step)
    env = textworld.gym.make(env_id)  # Create a Gym environment to play the text game.
    if verbose:
        if os.path.isdir(path):
            print(os.path.dirname(path), end="")
        else:
            print(os.path.basename(path), end="")

    # Collect some statistics: num_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores = [], [], []
    for no_episode in range(num_episodes):
        obs, infos = env.reset()  # Start new episode.

        score = 0
        done = False
        num_moves = 0
        while not done:
            command = agent.act(obs, score, done, infos)
            # print("observation:", obs)
            # print("command:", command)
            # print("score:", score)
            # input()
            obs, score, done, infos = env.step(command)
            num_moves += 1

        agent.act(obs, score, done, infos)  # Let the agent know the game is done.

        if verbose:
            print(".", end="")
        avg_moves.append(num_moves)
        avg_scores.append(score)
        avg_norm_scores.append(score / infos["max_score"])

    env.close()
    if verbose:
        if os.path.isdir(path):
            msg = "  \tavg. steps: {:5.1f}; avg. normalized score: {:4.1f} / {}."
            print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
        else:
            msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))

def train_multiple_games(agent, checkpoint_name, rewards="dense", num_episodes_per_game=5, verbose=True):

    print("Training on 100 games")
    agent.train()
    starttime = time()
    if rewards != "all":
        play(agent, f"./training_games_{rewards}/", num_episodes=100 * num_episodes_per_game, verbose=verbose)
    else:
        play(agent, f"./training_games_dense/", num_episodes=100 * num_episodes_per_game, verbose=verbose)
        play(agent, f"./training_games_balanced/", num_episodes=100 * num_episodes_per_game, verbose=verbose)
        play(agent, f"./training_games_sparse/", num_episodes=100 * num_episodes_per_game, verbose=verbose)
    print("Trained in {:.2f} secs".format(time() - starttime))

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(agent, f'checkpoints/{checkpoint_name}_{rewards}.pt')

def train_single_game(agent, checkpoint_name, reward_type="dense", num_episodes=500, verbose=True):
    print("*" * 25)
    print("Training on single game")
    agent.train() 
    starttime = time()
    play(agent, f"./games/tw-rewards{reward_type.title()}_goalDetailed.z8", num_episodes=num_episodes, verbose=verbose)
    print("Trained in {:.2f} secs".format(time() - starttime))
    print("*" * 50)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(agent, f'checkpoints/{checkpoint_name}.pt')

def test_multiple_games(checkpoint_name=None):
    if not checkpoint_name:
        agent = RandomAgent()
    else:
        agent = torch.load(f'checkpoints/{checkpoint_name}.pt')
    agent.test()
    play(agent, "./testing_games/", num_episodes=20 * 10)

def test_single_agent(checkpoint_name=None):
    if not checkpoint_name:
        agent = RandomAgent()
    else:
        agent = torch.load(f'checkpoints/{checkpoint_name}.pt')
        agent.test()
    play(agent, "./testing_games_dense/", num_episodes=20 * 10)
    play(agent, "./testing_games_balanced/", num_episodes=20 * 10)
    play(agent, "./testing_games_sparse/", num_episodes=20 * 10)
    # play(agent, "./games/tw-rewardsDense_goalDetailed.z8")  # Dense rewards game (training game for single)
    # play(agent, "./games/tw-rewardsBalanced_goalDetailed.z8")
    # play(agent, "./games/tw-rewardsSparse_goalDetailed.z8")
    # play(agent, "./games/tw-rewardsDense_goalBrief.z8")  # Dense rewards game (training game for single)
    # play(agent, "./games/tw-rewardsBalanced_goalBrief.z8")
    # play(agent, "./games/tw-rewardsSparse_goalBrief.z8")

def list_files(directory):
    filenames = os.listdir(directory)
    files_only = [f for f in filenames if os.path.isfile(os.path.join(directory, f))]
    return files_only

def test_random():
    agent = RandomAgent()
    print("*** TESTING RANDOM ***")
    play(agent, "./games/tw-rewardsDense_goalDetailed.z8")  # Dense rewards game
    play(agent, "./games/tw-another_game.z8")
    print("*" * 50)
    print()

def test_all_checkpoints():
    test_random()
    directory_path = './checkpoints'
    files = list_files(directory_path)

    for file in files:
        checkpoint_name = f"checkpoints/{file}"
        print("*** TESTING", file, "***")
        agent = torch.load(checkpoint_name)
        agent.test()
        # play(agent, "./games/tw-rewardsDense_goalDetailed.z8")  # Dense rewards game (training game for single)
        # play(agent, "./games/tw-rewardsBalanced_goalDetailed.z8")
        # play(agent, "./games/tw-rewardsSparse_goalDetailed.z8")
        # play(agent, "./games/tw-rewardsDense_goalBrief.z8")  # Dense rewards game (training game for single)
        # play(agent, "./games/tw-rewardsBalanced_goalBrief.z8")
        # play(agent, "./games/tw-rewardsSparse_goalBrief.z8")
        # play(agent, "./games/tw-another_game.z8") # Guaranteed to be different from the one above
        # play(agent, "./testing_games/", num_episodes=20 * 10)  # Averaged over 10 playthroughs for each test game

        play(agent, "./testing_games_dense/", num_episodes=20 * 10)
        play(agent, "./testing_games_balanced/", num_episodes=20 * 10)
        play(agent, "./testing_games_sparse/", num_episodes=20 * 10)

        print("*" * 50)
        print()

if __name__ == "__main__":
    # agent = RandomAgent()

    # evaluate random agent
    # play(agent, "./games/tw-rewardsDense_goalDetailed.z8")    # Dense rewards
    # play(agent, "./games/tw-rewardsBalanced_goalDetailed.z8") # Balanced rewards
    # play(agent, "./games/tw-rewardsSparse_goalDetailed.z8")   # Sparse rewards  
    # agent = NeuralAgentLSTM()
    # train_multiple_games(agent,"neural_agent_lstm_trained_on_multiple_games_2",  num_episodes_per_game=2)

    # agent = NeuralAgent(pretrained_embeddings=False)
    # train_multiple_games(agent,"neural_agent_trained_on_multiple_games_2",  num_episodes_per_game=2)
    # train_single_game()

    # agent = TransformerNeuralAgent()
    # train_single_game(agent, "transformer_neural_agent_single", 50)

    # test_single_agent("neural_agent_trained_on_multiple_games_2")

    # test_random()
    # test_all_checkpoints()

    # agent = NeuralAgent()
    # train_multiple_games(agent, "training_a2c_multiple_5", "all")
    # test_all_checkpoints()
    # test_single_agent('a2c_multiple_5_all')
    test_single_agent()
