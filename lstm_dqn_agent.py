import numpy as np
from copy import deepcopy
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import re
import random
from textworld.core import EnvInfos

"""
The ollowing code is from https://github.com/kolbytn/text-rl/tree/master Cheers to them! =)
We modified to fit with our GRU-A2C agent. 
"""

class ExperienceDataset(Dataset):
    def __init__(self, experience, keys):
        super(ExperienceDataset, self).__init__()
        self._keys = keys
        self._exp = []
        for x in experience:
            self._exp.append(x)

    def __getitem__(self, index):
        chosen_exp = self._exp[index]
        return tuple(chosen_exp[k] for k in self._keys)

    def __len__(self):
        return len(self._exp)

def dqn_collate(batch):

    transposed = {
        'reward': [],
        'terminal': [],
        'state_prime': [],
        'commands_prime': [],
        'hidden_prime': [],
        'state': [],
        'commands': [],
        'actions': [],
        'hidden': []
    }

    for step in batch:
        transposed['reward'].append(step[0])
        transposed['terminal'].append(step[7])
        transposed['state_prime'].append(step[1])
        transposed['commands_prime'].append(step[2])
        transposed['hidden_prime'].append(step[3])
        transposed['state'].append(step[4])
        transposed['commands'].append(step[5])
        transposed['actions'].append(step[6])
        transposed['hidden'].append(step[8])

    transposed['reward'] = torch.FloatTensor(transposed['reward'])
    transposed['terminal'] = torch.FloatTensor(np.array(transposed['terminal']))
    transposed['actions'] = torch.LongTensor(np.array(transposed['actions']))

    return transposed

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.length = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplay:
    def __init__(self, capacity, keys, epsilon=.01, alpha=.6):
        self._tree = SumTree(capacity)
        self.keys = keys
        self.e = epsilon
        self.a = alpha
        self.max_error = 5

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def extend(self, rollout):
        for step in rollout:
            self._tree.add(self.max_error, step)

    def sample(self, n):
        sample = []
        n = min(self._tree.length, n)
        segment = self._tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self._tree.get(s)
            data["idx"] = idx
            sample.append(data)
        return ExperienceDataset(sample, self.keys)

    def update(self, idx, error):
        self.max_error = max(self.max_error, error)
        p = self._get_priority(error)
        self._tree.update(idx, p)

class RecurrentNetReal(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu", use_last_action=False):
        super(RecurrentNetReal, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.obs_encoder_lstm = nn.LSTM(hidden_size, hidden_size)
        self.cmd_encoder_lstm = nn.LSTM(hidden_size, hidden_size)
        self.state_lstm = nn.LSTM(hidden_size, hidden_size)

        input_size = 2 * hidden_size
        self.dqn = nn.Sequential(
            nn.Linear(input_size, int(input_size * .75)),
            nn.ReLU(),
            nn.Linear(int(input_size * .75), int(input_size * .5)),
            nn.ReLU(),
            nn.Linear(int(input_size * .5), int(input_size * .25)),
            nn.ReLU(),
            nn.Linear(int(input_size * .25), 1)
        )

        self.hidden_size = hidden_size
        self.state_hidden = (torch.zeros(1, 1, hidden_size, device=device),
                             torch.zeros(1, 1, hidden_size, device=device))  # LSTM requires a tuple (h, c)
        self.last_cmds = None
        self.last_action = None
        self.device = device
        self.use_last_action = use_last_action

    def forward(self, obs, commands):
        nb_cmds = commands.size(1)

        if self.last_cmds is not None and self.use_last_action:
            obs = torch.cat((self.last_cmds[:, self.last_action], obs), dim=0)
        obs_embedding = self.embedding(obs)
        obs_encoder_hidden = self.obs_encoder_lstm(obs_embedding)[1]  # LSTM returns (output, (h, c))
        cmds_embedding = self.embedding(commands)
        cmds_encoder_hidden = self.cmd_encoder_lstm(cmds_embedding)[1]

        state_hidden = self.state_lstm(obs_encoder_hidden[0], self.state_hidden)[1]
        self.state_hidden = state_hidden

        state_input = torch.stack([state_hidden[0]] * nb_cmds, 2)
        dqn_input = torch.cat((state_input, cmds_encoder_hidden[0].unsqueeze(0)), dim=-1)

        scores = self.dqn(dqn_input).squeeze(-1)

        self.last_cmds = commands
        return scores

    def reset_hidden(self, batch_size):
        self.state_hidden = (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
                             torch.zeros(1, batch_size, self.hidden_size, device=self.device))
        self.last_cmds = None
        self.last_action = None

    def get_hidden(self):
        c = self.last_cmds.detach().cpu().numpy() if self.last_cmds is not None else None
        return [self.state_hidden[0].detach().cpu().numpy(), self.state_hidden[1].detach().cpu().numpy(), c, self.last_action]

    def set_hidden(self, tensors):
        self.state_hidden = (torch.from_numpy(tensors[0]).to(self.device),
                             torch.from_numpy(tensors[1]).to(self.device))
        self.last_cmds = torch.from_numpy(tensors[2]).to(self.device) if tensors[2] is not None else None
        self.last_action = tensors[3]

class LSTMDQN_AgentReal:
    def __init__(self, device="cpu", state_type="recurrent", train_freq=100, training_epochs=1,
                 batch_size=128, target_update=1000, sample_size=1000, lr=1e-3, eps=.99,
                 eps_decay=.9999, gamma=.9, max_vocab=1000, max_memory=30):
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.device = device

        # if state_type == "naive":
        #     self.model = NaiveNet(max_vocab, 128, device=device, use_last_action=True).to(device)
        # elif state_type == "recurrent":
        self.model = RecurrentNetReal(max_vocab, 128, device=device, use_last_action=True).to(device)
        # elif state_type == "memory":
        #     self.model = MemoryNet(max_vocab, 128, max_mem_size=max_memory, device=device, neighbors=True, use_last_action=True).to(device)
        # else:
        #     print("Invalid state type", state_type)
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.value_objective = nn.functional.smooth_l1_loss
        self.target = deepcopy(self.model).to(device)

        self.mode = "test"
        self.transitions = []
        self.replay_buffer = PrioritizedReplay(25, ('reward', 'state_prime', 'commands_prime', 'hidden_prime',
                                                         'state', 'commands', 'actions', 'terminal', 'hidden'))
        self.no_train_step = 0
        self.train_freq = train_freq
        self.training_epochs = training_epochs
        self.target_update = target_update
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.eps = eps
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.max_vocab = max_vocab
        self.wordcounts = {}

    def train(self):
        self.mode = "train"
        self.transitions = []
        self.model.reset_hidden(1)
        self.no_train_step = 0

    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)

    def act(self, obs, reward, done, infos):
        # Tokenize and pad the input and the commands to chose from.

        state = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"]) 

        input_tensor = self._prepare_tensor([state])
        commands_tensor = self._prepare_tensor(infos["admissible_commands"])

        # Get our next action.
        outputs = self.model(input_tensor, commands_tensor)
        self.eps *= self.eps_decay
        if np.random.random() > self.eps:
            index = torch.argmax(outputs).unsqueeze(0)
        else:
            index = torch.randint(len(infos["admissible_commands"]), (1,))
        self.model.last_action = index
        action = infos["admissible_commands"][index]

        # If testing finish
        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action

        # Else record transitions
        self.no_train_step += 1

        # Train
        if self.no_train_step % self.train_freq == 0:
            dataset = self.replay_buffer.sample(self.sample_size)
            if len(dataset) > 0:
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False,
                                    collate_fn=dqn_collate)
                self._train(loader)
                self.model.reset_hidden(1)

        if self.no_train_step % self.target_update == 0:
            self.target.load_state_dict(self.model.state_dict())

        if self.transitions:
            self.transitions[-1]['reward'] = reward  # Update reward information.
            self.transitions[-1]['terminal'] = np.array([int(done)])  # Update terminal information.
            self.transitions[-1]['state_prime'] = input_tensor.cpu().detach().numpy()  # Update state information.
            self.transitions[-1]['commands_prime'] = commands_tensor.cpu().detach().numpy()  # Update action information.
            self.transitions[-1]['hidden_prime'] = self.model.get_hidden()  # Update hidden state information.

        if done:
            self.replay_buffer.extend(self.transitions)
            self.transitions = []
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            # Reward will be set on the next call
            self.transitions.append({'reward': None,  # Reward
                                     'terminal': None,  # Terminal
                                     'state_prime': None,  # Next State
                                     'commands_prime': None,  # Next Actions
                                     'hidden_prime': None,  # Next Hidden State
                                     'state': input_tensor.cpu().detach().numpy(),  # State
                                     'commands': commands_tensor.cpu().detach().numpy(),  # Actions
                                     'actions': index.cpu().detach().numpy(),  # Chosen Actions
                                     'hidden': self.model.get_hidden()})  # Hidden state of model
        return action

    def _train(self, loader):
        for epoch in range(self.training_epochs):
            for batch, step in enumerate(loader):
                self.optimizer.zero_grad()

                reward = step['reward'].to(self.device).unsqueeze(-1)
                terminal = step['terminal'].to(self.device)
                actions = step['actions'].to(self.device)

                model_q = self._get_model_out(self.model, step['state'], step['commands'], step['hidden'], indices=actions)
                target_out = self._get_model_out(self.target, step['state_prime'], step['commands_prime'], step['hidden_prime'])
                target_q = reward + self.gamma * target_out * (1 - terminal)

                loss = torch.mean(self.value_objective(target_q, model_q))
                loss.backward()
                self.optimizer.step()

    def _get_model_out(self, model, state, commands, hidden, indices=None):
        q_values = []
        for i in range(len(hidden)):
            s = torch.from_numpy(state[i]).to(self.device)
            c = torch.from_numpy(commands[i]).to(self.device)
            model.set_hidden(hidden[i])
            q = model(s, c)
            if indices is None:
                q = torch.max(q, 2)[0].squeeze(-1)
            else:
                q = q[0, 0, indices[i]]
            q_values.append(q)
        return torch.stack(q_values)

    def _get_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.max_vocab:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)
            self.wordcounts[word] = 0
        self.wordcounts[word] += 1
        return self.word2id[word]

    def _get_token(self, text):
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_id, text.split()))
        return word_ids

    def _prepare_tensor(self, texts):
        texts = list(map(self._get_token, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
        padded_tensor = padded_tensor.permute(1, 0)
        return padded_tensor
