import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT  model
class DNN(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=2, num_features_xt=954, dropout=0.2):
        super(DNN, self).__init__()

        # graph drug1 layers

        self.drug1_fc_g1 = nn.Linear(num_features_xd * 2 + num_features_xt, 4096)
        self.drug1_fc_g2 = nn.Linear(4096, 1024)
        self.drug1_fc_g3 = nn.Linear(1024, 256)
        self.drug1_fc_g4 = nn.Linear(256, n_output)






        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x1 = F.normalize(inputs, 2, 1)
        # deal drug1
        x1 = self.drug1_fc_g1(x1)
        x1 = torch.relu(x1)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.drug1_fc_g2(x1)
        x1 = torch.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.drug1_fc_g3(x1)
        x1 = torch.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        out = self.drug1_fc_g4(x1)
        return out
