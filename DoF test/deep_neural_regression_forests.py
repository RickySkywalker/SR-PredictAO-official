import torch
import torch as th
from torch import nn
import numpy as np
import deep_nerual_regression_tree

class NeuralRegressionForest(nn.Module):

    def __init__(
            self,
            num_trees,
            depth,
            num_features,
            used_features_rate,
            num_classes,
            dropout_rate,
            device=th.device('cuda' if th.cuda.is_available() else 'cpu')
    ):
        super(NeuralRegressionForest, self).__init__()
        self.ensemble = nn.ModuleList()
        self.num_classes = num_classes
        # self.device = device
        self.device = device
        self.num_trees = num_trees

        # CNN layer, addition for decision function
        # self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=3)
        # self.num_features = int((num_features-2)/3)

        self.num_features = num_features

        for i in range(num_trees):
            self.ensemble.append(
                deep_nerual_regression_tree.NeuralRegressionTree(
                    depth,
                    self.num_features,
                    used_features_rate,
                    num_classes,
                    dropout_rate,
                    device=device
                )
            )

    def forward(self, features):
        
        batch_size = features.size()[0]
        outputs = th.zeros((batch_size, self.num_classes), device=self.device)
        
        # # An optional CNN layer, an addition layer for decision function
        # features = th.reshape(features, (batch_size, 1, -1))
        # features = self.conv(features)
        # features = th.reshape(features, (batch_size, -1))

        for i, tree in enumerate(self.ensemble):
            prob = tree(features)
            outputs += prob
        outputs /= len(self.ensemble)

        return outputs

