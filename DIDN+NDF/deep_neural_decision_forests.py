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


        self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=3)
        self.num_features = int((num_features-3)/3) + 1

        for i in range(num_trees):
            curr_device = th.device('cuda' if th.cuda.is_available() else 'cpu')
            self.ensemble.append(
                deep_nerual_decision_tree.NeuralDecisionTree(
                    depth,
                    self.num_features,
                    used_features_rate,
                    num_classes,
                    device=device
                )
            )

    def forward(self, features):


        batch_size = features.size()[0]
        outputs = th.zeros((batch_size, self.num_classes), device=self.device)
        features = th.reshape(features, (batch_size, 1, -1))
        features = self.conv(features)
        features = th.reshape(features, (batch_size, -1))

        for i, tree in enumerate(self.ensemble):
            prob = tree(features)
            outputs += prob
        outputs /= len(self.ensemble)
        return outputs

    def RF15(self, base_model_result, RF_result):
        base_model_var = th.exp(th.var(base_model_result, dim=1).reshape((-1, 1)))
        RF_var = th.exp(th.var(RF_result, dim=1).reshape(-1, 1))

        score = (base_model_result * base_model_var + RF_result * RF_var) / (base_model_var + RF_var)

        return score