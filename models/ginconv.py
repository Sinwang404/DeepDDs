import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=2, num_features_xd=78, num_features_xt=954,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # convolution drug1 layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.drug1_conv1 = GINConv(nn1)
        self.drug1_bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug1_conv2 = GINConv(nn2)
        self.drug1_bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug1_conv3 = GINConv(nn3)
        self.drug1_bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug1_conv4 = GINConv(nn4)
        self.drug1_bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug1_conv5 = GINConv(nn5)
        self.drug1_bn5 = torch.nn.BatchNorm1d(dim)

        self.drug1_fc1_xd = Linear(dim, output_dim)

        # convolution drug2 layers
        nn6 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.drug2_conv1 = GINConv(nn6)
        self.drug2_bn1 = torch.nn.BatchNorm1d(dim)

        nn7 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug2_conv2 = GINConv(nn7)
        self.drug2_bn2 = torch.nn.BatchNorm1d(dim)

        nn8 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug2_conv3 = GINConv(nn8)
        self.drug2_bn3 = torch.nn.BatchNorm1d(dim)

        nn9 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug2_conv4 = GINConv(nn9)
        self.drug2_bn4 = torch.nn.BatchNorm1d(dim)

        nn10 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.drug2_conv5 = GINConv(nn10)
        self.drug2_bn5 = torch.nn.BatchNorm1d(dim)

        self.drug2_fc1_xd = Linear(dim, output_dim)

        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )


        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(output_dim * 3, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1 = F.relu(self.drug1_conv1(x1, edge_index1))
        x1 = self.drug1_bn1(x1)
        x1 = F.relu(self.drug1_conv2(x1, edge_index1))
        x1 = self.drug1_bn2(x1)
        x1 = F.relu(self.drug1_conv3(x1, edge_index1))
        x1 = self.drug1_bn3(x1)
        x1 = F.relu(self.drug1_conv4(x1, edge_index1))
        x1 = self.drug1_bn4(x1)
        x1 = F.relu(self.drug1_conv5(x1, edge_index1))
        x1 = self.drug1_bn5(x1)
        x1 = global_add_pool(x1, batch1)
        x1 = F.relu(self.drug1_fc1_xd(x1))
        x1 = F.dropout(x1, p=0.2, training=self.training)

        # deal drug2
        x2 = F.relu(self.drug1_conv1(x2, edge_index2))
        x2 = self.drug1_bn1(x2)
        x2 = F.relu(self.drug1_conv2(x2, edge_index2))
        x2 = self.drug1_bn2(x2)
        x2 = F.relu(self.drug1_conv3(x2, edge_index2))
        x2 = self.drug1_bn3(x2)
        x2 = F.relu(self.drug1_conv4(x2, edge_index2))
        x2 = self.drug1_bn4(x2)
        x2 = F.relu(self.drug1_conv5(x2, edge_index2))
        x2 = self.drug1_bn5(x2)
        x2 = global_add_pool(x2, batch2)
        x2 = F.relu(self.drug1_fc1_xd(x2))
        x2 = F.dropout(x2, p=0.2, training=self.training)

        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)

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
