from neural_agent import NeuralAgent
from random_agent import RandomAgent
from lstmdqn_agent import NeuralAgent_LSTMDQN
import os
from glob import glob
from typing import Mapping, Any
import numpy as np
import textworld.gym
import torch
from time import time


def play(agent, path, max_step=100, nb_episodes=10, verbose=True):
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

    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores = [], [], []
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.

        score = 0
        done = False
        nb_moves = 0
        while not done:
            command = agent.act(obs, score, done, infos)
            obs, score, done, infos = env.step(command)
            nb_moves += 1

        agent.act(obs, score, done, infos)  # Let the agent know the game is done.

        if verbose:
            print(".", end="")
        avg_moves.append(nb_moves)
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

def train(agent, training_path, nb_episodes, verbose=False):
    # This function should encapsulate the training logic for agents.
    pass

# Example usage:
if __name__ == "__main__":
    agent = RandomAgent()

    # evaluate random agent
    # play(agent, "./games/tw-rewardsDense_goalDetailed.z8")    # Dense rewards
    # play(agent, "./games/tw-rewardsBalanced_goalDetailed.z8") # Balanced rewards
    # play(agent, "./games/tw-rewardsSparse_goalDetailed.z8")   # Sparse rewards  
    
    agent = NeuralAgent_LSTMDQN()

    print("Training")
    agent.train()  # Tell the agent it should update its parameters.
    starttime = time()
    play(agent, "./games/tw-rewardsDense_goalDetailed.z8", nb_episodes=500, verbose=False)  # Dense rewards game.

    print("Trained in {:.2f} secs".format(time() - starttime))

    # Save the trained agent.
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(agent, 'checkpoints/lstmdqn_agent_trained_on_single_game.pt')
