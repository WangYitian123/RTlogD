import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax


class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, M1)
            Previous edge features.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    """Generate context for each node (atom) by message passing at the beginning.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        """Edge feature update."""
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update."""
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.
        edge_feats : float32 tensor of shape (E, N2)
            Input edge features. E for the number of edges and N2 for the feature size.

        Returns
        -------
        float32 tensor of shape (V, N3)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):
    """GNNLayer for updating node features.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the input graph features.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def apply_edges(self, edges):
        """Edge feature update by concatenating the features of the destination
        and source nodes."""
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.

        Returns
        -------
        float32 tensor of shape (V, N1)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)


class Pka_acidic(nn.Module):

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 graph_feat_size,
                 output_size,
                 dropout):
        super(Pka_acidic,self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):

        mask = torch.sum(g.ndata['h'][:,-4:],dim = 1) * (1 - g.ndata['h'][:,0])
        mask = 1/mask -1
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        atom_pka = self.predict(node_feats)
        atom_pka = -(atom_pka + mask.reshape(-1,1))
        g.ndata['h'] = torch.pow(10,atom_pka)
        g_feats = -torch.log10(dgl.sum_nodes(g, 'h')+10**-38)
        return g_feats


class Pka_basic(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 graph_feat_size,
                 output_size,
                 dropout):
        super(Pka_basic, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        mask = g.ndata['h'][:,1] * (1 - g.ndata['h'][:,61])
        mask = -1/mask +1
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        atom_pka = self.predict(node_feats)
        atom_pka = (atom_pka + mask.reshape(-1,1))
        g.ndata['h'] = torch.pow(10,atom_pka)
        g_feats = torch.log10(dgl.sum_nodes(g, 'h'))
        return g_feats


class Pka_acidic_view(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 graph_feat_size,
                 output_size,
                 dropout):
        super(Pka_acidic_view,self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):

        mask = torch.sum(g.ndata['h'][:,-4:],dim = 1) * (1 - g.ndata['h'][:,0])
        mask = 1/mask -1
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        atom_pka = self.predict(node_feats)
        atom_pka = -(atom_pka + mask.reshape(-1,1))
        g.ndata['h'] = torch.pow(10,atom_pka)
        g_feats = -torch.log10(dgl.sum_nodes(g, 'h'))
        atom_pka_out = atom_pka * -1
        atom_pka_out = torch.squeeze(atom_pka_out)
        
        return g_feats, atom_pka_out.detach().cpu().numpy().tolist()


class Pka_basic_view(nn.Module):

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 graph_feat_size,
                 output_size,
                 dropout):
        super(Pka_basic_view,self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):

        mask = g.ndata['h'][:,1] * (1 - g.ndata['h'][:,61])
        #print(mask)
        mask = -1/mask +1
        #print(mask)
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        atom_pka = self.predict(node_feats)
        #print(atom_pka)
        atom_pka = (atom_pka + mask.reshape(-1,1))
        #print(atom_pka)
        g.ndata['h'] = torch.pow(10,atom_pka)
        g_feats = torch.log10(dgl.sum_nodes(g, 'h'))
        atom_pka_out = torch.squeeze(atom_pka)
        
        return g_feats, atom_pka_out.detach().cpu().numpy().tolist()