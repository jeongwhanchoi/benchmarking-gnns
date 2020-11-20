import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer_convex import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']   
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # cvx_layer
        self.layers_cvx = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual, method='convex')])
        self.layers_cvx.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, method='normal'))
        self.layers_cvx.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, method='normal'))
        self.layers_cvx.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, method='normal'))
        self.MLP_layer_cvx = MLPReadout(out_dim, n_classes)
        
        # cov_layer
        self.layers_cov = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual, method='concave')])
        self.layers_cov.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, method='normal'))
        self.layers_cov.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, method='normal'))
        self.layers_cov.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, method='normal'))
        self.MLP_layer_cov = MLPReadout(out_dim, n_classes)

        self.layers_normal = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual, method='normal') for _ in range(n_layers-1)])
        self.layers_normal.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, method='normal'))
        self.MLP_layer_normal = MLPReadout(out_dim, n_classes)

        # filter 
        # self.cvx_filter = Variable(torch.ones(15295).type(torch.FloatTensor), requires_grad=True).unsqueeze(1) #15295
        self.cvx_filter = Variable(torch.ones(15279).type(torch.FloatTensor), requires_grad=True).unsqueeze(1) #15279
        # self.cvx_filter = Variable(torch.randn(15139).type(torch.FloatTensor), requires_grad=True).unsqueeze(1)
        # self.cvx_filter = Variable(torch.randn(15035).type(torch.FloatTensor), requires_grad=True).unsqueeze(1)
        # self.cvx_filter = Variable(torch.randn(15327).type(torch.FloatTensor), requires_grad=True).unsqueeze(1)
        self.cov_filter = Variable(torch.ones(15279).type(torch.FloatTensor), requires_grad=True).unsqueeze(1)#15279
        # self.cov_filter = Variable(torch.randn(15139).type(torch.FloatTensor), requires_grad=True).unsqueeze(1)
        self.normal_filter = Variable(torch.ones(15279).type(torch.FloatTensor), requires_grad=True).unsqueeze(1)#15279
        # self.normal_filter = Variable(torch.randn(15139).type(torch.FloatTensor), requires_grad=True).unsqueeze(1)

    def forward(self, g, h, e):
        # import pdb
        # pdb.set_trace()
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        # h = h.to(self.device)
        
        # convex part
        cvx_filter = torch.sigmoid(self.cvx_filter)
        cvx_filter = cvx_filter.to(device=self.device)
        cvx_h = h * cvx_filter

        for conv in self.layers_cvx:
            h = conv(g,cvx_h)
        
        h_out_cvx = self.MLP_layer_cvx(h)
        
        # concave part
        cov_filter = torch.sigmoid(self.cov_filter)
        cov_filter = cov_filter.to(device=self.device)
        cov_h = h * cov_filter

        for conv in self.layers_cov:
            h = conv(g,cov_h)
        
        h_out_cov = self.MLP_layer_cov(h)

        # normal part
        normal_filter = torch.sigmoid(self.normal_filter)
        normal_filter = normal_filter.to(device=self.device)
        normal_h = h * normal_filter

        for conv in self.layers_normal:
            h = conv(g,normal_h)
        
        h_out_normal = self.MLP_layer_normal(h)

        h_out = torch.cat([h_out_cvx, h_out_cov, h_out_normal])#TODO: ???
        # graphsage
        #for conv in self.layers:
        #    h = conv(g, h)
        # output
        #h_out = self.MLP_layer(h)

        return h_out
    

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss