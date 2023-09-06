import torch
import torch as th
from torch import nn
import numpy as np
import deep_nerual_decision_tree

class NeuralDecisionForest(nn.Module):

    def __init__(
            self,
            num_trees,
            depth,
            num_features,
            used_features_rate,
            num_classes,
            device=th.device('cuda' if th.cuda.is_available() else 'cpu')
    ):

        super(NeuralDecisionForest, self).__init__()
        self.ensemble = nn.ModuleList()
        self.num_classes = num_classes
        self.device = device
        self.num_trees = num_trees

        for i in range(num_trees):
            self.ensemble.append(
                deep_nerual_decision_tree.NeuralDecisionTree(
                    depth,
                    num_features,
                    used_features_rate,
                    num_classes,
                    device=device
                )
            )

    def forward(self, features):

        batch_size = features.size()[0]
        outputs = th.zeros((batch_size, self.num_classes), device=self.device)

        for i, tree in enumerate(self.ensemble):
            prob = tree(features)
            outputs += prob
        outputs /= len(self.ensemble)
        return outputs