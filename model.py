import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.base = MLPBase(obs_shape[0])
       

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.hidden_size, num_outputs)
        else:
            raise NotImplementedError


    def forward(self, inputs, masks):
        raise NotImplementedError

    def act(self, inputs, masks, deterministic=False):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs, masks)
        return value

    def evaluate_actions(self, inputs, masks, action):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


# class NNBase(nn.Module):

#     def __init__(self, recurrent, recurrent_input_size, hidden_size):
#         super(NNBase, self).__init__()

#         self._hidden_size = hidden_size
#         self._recurrent = recurrent

#         if recurrent:
#             self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
#             nn.init.orthogonal_(self.gru.weight_ih.data)
#             nn.init.orthogonal_(self.gru.weight_hh.data)
#             self.gru.bias_ih.data.fill_(0)
#             self.gru.bias_hh.data.fill_(0)

#     @property
#     def is_recurrent(self):
#         return self._recurrent

#     @property
#     def recurrent_hidden_state_size(self):
#         if self._recurrent:
#             return self._hidden_size
#         return 1

#     @property
#     def output_size(self):
#         return self._hidden_size

#     def _forward_gru(self, x, hxs, masks):
#         if x.size(0) == hxs.size(0):
#             x = hxs = self.gru(x, hxs * masks)
#         else:
#             # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
#             N = hxs.size(0)
#             T = int(x.size(0) / N)

#             # unflatten
#             x = x.view(T, N, x.size(1))

#             # Same deal with masks
#             masks = masks.view(T, N, 1)

#             outputs = []
#             for i in range(T):
#                 hx = hxs = self.gru(x[i], hxs * masks[i])
#                 outputs.append(hx)

#             # assert len(outputs) == T
#             # x is a (T, N, -1) tensor
#             x = torch.stack(outputs, dim=0)
#             # flatten
#             x = x.view(T * N, -1)

#         return x, hxs

class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        # super(nn.Modu, self).__init__(recurrent, num_inputs, hidden_size)

        nn.Module.__init__(self)

        self.hidden_size = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor
