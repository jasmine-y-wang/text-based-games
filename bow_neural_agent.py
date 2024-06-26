import re
from typing import List, Mapping, Any, Optional
from collections import defaultdict

import numpy as np

import textworld
import textworld.gym
import textworld.core
from textworld.core import EnvInfos

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torchtext.vocab import GloVe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CommandScorerBOW(nn.Module):
    def __init__(self, input_size, hidden_size=50, pretrained_embeddings=False):
        super(CommandScorer_BOW, self).__init__()
        if pretrained_embeddings:
            glove_weights = torch.load(f".vector_cache/glove.6B.{hidden_size}d.txt.pt")
            self.embedding = nn.Embedding.from_pretrained(glove_weights[2], freeze=False)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)
        
        self.hidden_size  = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.critic = nn.Linear(hidden_size, 1)
        self.att_cmd = nn.Linear(hidden_size * 2, 1)

    def forward(self, obs, commands):
        embedded_obs = self.embedding(obs).mean(dim=0)  # Shape: [hidden_size]
        embedded_cmds = self.embedding(commands).mean(dim=1)  # Shape: [num_commands, hidden_size]

        value = self.critic(embedded_obs.unsqueeze(0))  # Unsqueeze to add batch dimension

        # Ensure that embedded_obs is correctly shaped and expanded
        state_expanded = embedded_obs.unsqueeze(0).expand(embedded_cmds.size(0), embedded_obs.size(0))  # Corrected expansion

        cmd_selector_input = torch.cat([state_expanded, embedded_cmds], dim=1)  # Concatenate along the last dimension
        
        scores = self.att_cmd(cmd_selector_input)
        scores = scores.squeeze(-1)  # Flatten the scores if necessary
        probs = F.softmax(scores, dim=0)
        index = probs.multinomial(num_samples=1).item()

        return scores, index, value
    
    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)

class NeuralAgentBOW(textworld.core.Agent):
    MAX_VOCAB_SIZE = 1000
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9

    def __init__(self, pretrained_embeddings=False):
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.model = CommandScorerBOW(input_size=self.MAX_VOCAB_SIZE, hidden_size=100, pretrained_embeddings=pretrained_embeddings)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)
        self.mode = "test"

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)
    
    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0

    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)

    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) < self.MAX_VOCAB_SIZE:
                self.id2word.append(word)
                self.word2id[word] = len(self.word2id) - 1
            return self.word2id["<UNK>"]
        return self.word2id[word]

    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0) # Batch x Seq => Seq x Batch
        return padded_tensor
    
    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]


    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> Optional[str]:
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])

        # Get our next action and value prediction.
        outputs, indexes, values = self.model(input_tensor, commands_tensor)
        action = infos["admissible_commands"][indexes[0]]

        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action

        self.no_train_step += 1

        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
            self.last_score = score
            if infos["won"]:
                reward += 100
            if infos["lost"]:
                reward -= 100

            self.transitions[-1][0] = reward  # Update reward information.

        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)

            loss = 0
            for transition, ret, advantage in zip(self.transitions, returns, advantages):
                reward, indexes_, outputs_, values_ = transition

                advantage        = advantage.detach() # Block gradients flow here.
                probs            = F.softmax(outputs_, dim=2)
                log_probs        = torch.log(probs)
                log_action_probs = log_probs.gather(2, indexes_)
                policy_loss      = (-log_action_probs * advantage).sum()
                value_loss       = (.5 * (values_ - ret) ** 2.).sum()
                entropy     = (-probs * log_probs).sum()
                loss += policy_loss + 0.5 * value_loss - 0.1 * entropy

                self.stats["mean"]["reward"].append(reward)
                self.stats["mean"]["policy"].append(policy_loss.item())
                self.stats["mean"]["value"].append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())

            if self.no_train_step % self.LOG_FREQUENCY == 0:
                msg = "{:6d}. ".format(self.no_train_step)
                msg += "  ".join("{}: {: 3.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                msg += "  " + "  ".join("{}: {:2d}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                msg += "  vocab: {:3d}".format(len(self.id2word))
                print(msg)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.transitions = []
            self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call

        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.

        return action