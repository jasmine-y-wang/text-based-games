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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CommandScorerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CommandScorerLSTM, self).__init__()
        torch.manual_seed(42)  # For reproducibility
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size)  # LSTM for the encoder
        self.cmd_encoder_lstm = nn.LSTM(hidden_size, hidden_size)  # LSTM for command encoding
        self.state_lstm = nn.LSTM(hidden_size, hidden_size)  # LSTM for state updates
        self.hidden_size = hidden_size
        self.state_hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        self.critic = nn.Linear(hidden_size, 1)
        self.att_cmd = nn.Linear(hidden_size * 2, 1)

    def forward(self, obs, commands, **kwargs):
        input_length = obs.size(0)
        batch_size = obs.size(1)
        nb_cmds = commands.size(1)

        embedded = self.embedding(obs)
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder_lstm(embedded)
        state_output, (state_hidden, state_cell) = self.state_lstm(encoder_hidden, self.state_hidden)
        self.state_hidden = (state_hidden, state_cell)
        value = self.critic(state_output.squeeze(0))

        # Attention network over the commands.
        cmds_embedding = self.embedding(commands)
        _, (cmds_encoding_last_states, _) = self.cmd_encoder_lstm(cmds_embedding)  # 1 x cmds x hidden

        # Expand state representation for each command
        cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden

        # Expand command encoding for the entire batch
        cmds_encoding_last_states = torch.stack([cmds_encoding_last_states] * batch_size, 1)  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings
        cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1)

        # Compute one score per command
        scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x Batch x cmds

        probs = F.softmax(scores, dim=2)  # 1 x Batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0)  # 1 x batch x index
        optimal_index = torch.argmax(probs, dim=2).unsqueeze(0)
        is_optimal = (index == optimal_index).item()
        return scores, index, value, is_optimal

    def reset_hidden(self, batch_size):
        self.state_hidden = (torch.zeros(1, batch_size, self.hidden_size),
                             torch.zeros(1, batch_size, self.hidden_size))

class NeuralAgentLSTM:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 1000
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9

    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

        self.model = CommandScorerLSTM(input_size=self.MAX_VOCAB_SIZE, hidden_size=128)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)

        self.mode = "test"
        self.optimal_action_count = 0
        self.total_action_count = 0
        self.optimal_actions_log = {}

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

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)
    
    @property
    def embedding(self) -> nn.Embedding:
        return self.model.embedding

    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)
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

        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])

        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])

        # Get our next action and value prediction.
        outputs, indexes, values, bestAction = self.model(input_tensor, commands_tensor)
        action = infos["admissible_commands"][indexes[0]]

        # Increment total actions
        self.total_action_count += 1
        
        # Determine if selected action was optimal
        if bestAction:
            self.optimal_action_count += 1

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
                self.optimal_actions_log[self.no_train_step] = self.optimal_action_count / self.total_action_count
                print(self.optimal_action_count, self.total_action_count)

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

    def calculate_optimal_action_percentage(self):
        if self.total_action_count == 0:
            return 0
        return (self.optimal_action_count / self.total_action_count) * 100

    def reset_statistics(self):
        self.optimal_action_count = 0
        self.total_action_count = 0
        self.optimal_actions_log.clear()