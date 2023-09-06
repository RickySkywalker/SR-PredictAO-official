'''
Author: WANG, Ruida
'''
import torch
import torch as th
from torch import nn
import numpy as np

from torch.nn import functional

class NeuralDecisionTree(nn.Module):
    def __init__(self,
                 depth,
                 num_features,
                 used_features_rate,
                 num_classes,
                 cuda_gpu=0,
                 device=th.device('cuda' if th.cuda.is_available() else 'cpu')
        ):
        super(NeuralDecisionTree, self).__init__()

        self.cuda_gpu = cuda_gpu
        self.depth = depth
        self.num_leaves = 2**depth
        self.num_classes = num_classes

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]
        self.device = device

        # Initialize the weights of the classes in leaves
        self.pi = th.tensor(
            data=th.normal(0.0, 0.5, size=(self.num_leaves, self.num_classes)),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        self.decision_fn_1 = nn.Linear(num_used_features, int(num_used_features + self.num_leaves/2), device=self.device)
        self.activation_fn_1 = nn.Sigmoid()
        self.decision_fn_2 = nn.Linear(int(num_used_features + self.num_leaves/2),
                                       int(num_used_features + self.num_leaves/2),
                                       device=self.device)
        self.activation_fn_2 = nn.Tanh()
        self.decision_fn_3 = nn.Linear(int(num_used_features + self.num_leaves/2),
                                       self.num_leaves,
                                       device=self.device)
        self.activation_fn_3 = nn.Sigmoid()
        self.mu_dropout = nn.Dropout(0.4)


    def forward(self, features):
        batch_size = features.size()[0]

        temp = th.tensor(np.transpose(self.used_features_mask), dtype=th.float32, device=self.device)

        feat = th.matmul(
            features,
            temp,
        )

        feat = self.activation_fn_1(self.decision_fn_1(feat))
        feat = self.activation_fn_2(self.decision_fn_2(feat))
        feat = self.activation_fn_3(self.decision_fn_3(feat))

        decisions = th.unsqueeze(feat, dim=2)
        decisions = th.cat((decisions, 1-decisions), dim=2)

        mu = th.ones((batch_size, 1, 1), device=self.device)

        begin_idx = 1
        end_idx = 2

        for level in range(self.depth):
            mu = th.reshape(mu, (batch_size, -1, 1))
            mu = th.tile(mu, (1, 1, 2))
            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions
            begin_idx = end_idx
            end_idx = begin_idx + 2**(level + 1)

        mu = th.reshape(mu, (batch_size, self.num_leaves))
        mu = self.mu_dropout(mu)
        probabilities = th.softmax(self.pi, dim=0)
        outputs = th.matmul(mu, probabilities)
        return outputs
