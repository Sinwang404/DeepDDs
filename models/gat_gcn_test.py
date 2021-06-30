import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=2, num_features_xd=78, num_features_xt=954,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output

        self.drug1_conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.drug1_conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*10*2, 512)
        self.drug1_fc_g2 = torch.nn.Linear(512, 128)
        self.drug1_fc_g3 = torch.nn.Linear(128, output_dim)

        self.drug2_conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.drug2_conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 512)
        self.drug2_fc_g2 = torch.nn.Linear(512, 128)
        self.drug2_fc_g3 = torch.nn.Linear(128, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )


        # combined layers
        self.fc1 = nn.Linear(output_dim * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch


        # deal drug1
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x1 = torch.cat([gmp(x1, batch1), gap(x1, batch1)], dim=1)
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.drug1_fc_g3(x1)

        # deal drug2
        x2 = self.drug1_conv1(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.drug1_conv2(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x2 = torch.cat([gmp(x2, batch2), gap(x2, batch2)], dim=1)
        x2 = self.relu(self.drug1_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug1_fc_g2(x2)
        x2 = self.drug1_fc_g3(x2)


        # deal cell

        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)

        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
