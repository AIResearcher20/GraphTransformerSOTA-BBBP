import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm, global_add_pool, TransformerConv

class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_channels, heads=4, dropout=0.3):
        super().__init__()
        self.attn = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout, beta=True)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.attn(x, edge_index)
        x = self.norm1(x + self.dropout(h))
        h = self.ffn(x)
        x = self.norm2(x + self.dropout(h))
        return x

class GraphTransformerSOTA2026(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=4, dropout=0.3):
        super().__init__()
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.gnn_layers = nn.ModuleList()
        self.gnn_bns = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
            ))
            self.gnn_bns.append(BatchNorm(hidden_channels))
            self.trans_layers.append(GraphTransformerBlock(hidden_channels, heads, dropout))
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x.dtype == torch.long:
            x = x.float()
        x = self.encoder(x)
        for gnn, bn, trans in zip(self.gnn_layers, self.gnn_bns, self.trans_layers):
            x_gnn = gnn(x, edge_index)
            x_gnn = bn(x_gnn)
            x_gnn = F.relu(x_gnn)
            x_trans = trans(x, edge_index)
            g = torch.sigmoid(self.gate)
            x = g * x_gnn + (1 - g) * x_trans
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)

def create_model(config):
    torch.manual_seed(config['experiment']['seed'])
    return GraphTransformerSOTA2026(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels'],
        num_layers=config['model']['num_layers'],
        heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    )
