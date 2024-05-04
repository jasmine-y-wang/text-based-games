from neural_agent import NeuralAgent, CommandScorer
from random_agent import RandomAgent
from lstm_neural_agent import NeuralAgentLSTM
import os
from glob import glob
from typing import Mapping, Any
import numpy as np
import textworld.gym
import torch
from time import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_embeddings(checkpoint_name):
    agent = torch.load(f'checkpoints/{checkpoint_name}.pt')
    embedding = agent.embedding
    # Access the weights
    embedding_weights = embedding.weight.data.cpu().numpy()

    top_words = ['examine', 'door', 'old', 'key', 'chest', 'drawer', 'see', 'wooden', 'antique', 'trunk', 'king-size', 'bed', 'close', 'open', 'empty', 'put', 'take', 'go', 'table', 'leading', 'kitchen', 'carrying', 'screen', 'bbq', 'inventory', 'look', 'set', 'chairs', 'nothing', 'closed', 'look', 'island', 'refrigerator', 'east', 'south']

    top_indices = top_indices = [agent.word2id[word] for word in top_words if word in agent.word2id]

    top_embeddings = embedding_weights[top_indices]

    # Adjust perplexity if necessary (make sure it's less than the number of top words)
    perplexity_value = min(30, len(top_words) - 1)  # Subtract 1 to ensure it's strictly less

    # Initialize t-SNE, you can adjust parameters like `n_components`, `perplexity`, and `n_iter`
    tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=1000)

    # Fit and transform the data
    tsne_embeddings = tsne.fit_transform(top_embeddings)

    # Plot the results
    plt.figure(figsize=(10, 7))
    # plt.title('t-SNE of Embeddings')

    # LEGEND
    # for i, word in enumerate(top_words):
    #     plt.scatter(tsne_embeddings[i, 0], tsne_embeddings[i, 1], label=word)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()  # Adjusts plot to include the legend
    # plt.savefig(f'embeddings_legend_{checkpoint_name}.png')

    # ANNOTATED
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0.5)
    # Annotate the selected top words
    for i, word in enumerate(top_words):
        plt.annotate(" " + word, (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), fontsize=14)
    plt.savefig(f'embeddings/{checkpoint_name}_embeddings.png')

if __name__ == "__main__":
    get_embeddings("neural_agent_trained_on_multiple_games")
    get_embeddings("neural_agent_trained_on_multiple_games_pretrained_embeddings_2")
    get_embeddings("neural_agent_trained_on_single_game_50")
    get_embeddings("neural_agent_trained_on_single_game_pretrained_embeddings_50")
    get_embeddings("neural_agent_trained_on_single_game_500")
    get_embeddings("neural_agent_trained_on_single_game_pretrained_embeddings_500")