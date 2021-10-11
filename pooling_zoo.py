import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import SAGPooling, TopKPooling,EdgePooling,ASAPooling,dense_diff_pool,LEConv,GINConv,GraphConv,GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn.conv import MessagePassing
import torch
from torch.nn import Linear, Sequential, ReLU,ELU, BatchNorm1d as BN
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import reset
from torch_geometric.nn.pool.topk_pool import topk

class ASAPooling_mix(ASAPooling):

    def forward(self, x, edge_index, edge_weight=None, batch=None, add_self_loop=False, remove_self_loop=False, ft=False):
        N = x.size(0)

        if self.add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value=1)
                # edge_index, edge_weight, fill_value=1, num_nodes=N)
        if edge_weight==None:
            edge_weight = torch.ones(edge_index.size()[1], device=x.device)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_pool = x
        if self.GNN is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max') #Eq.5 in ASAP
        x_q = self.lin(x_q)[edge_index[1]]  #Wm_i in Eq.6

        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)  #W * \sigma(Wm_i || xj) in Eq.6
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N) #Eq.6

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training)

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='add') #Eq.7

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index, edge_weight=edge_weight).sigmoid().view(-1)
        perm = topk(fitness, self.ratio, batch)
        if ft:
            x = x[perm] * fitness[perm].view(-1, 1)
            batch = batch[perm]
            # edge_index, edge_attr = filter_adj(edge_index, None, perm,
            #                                    num_nodes=score.size(0))
            # Graph coarsening.
            row, col = edge_index
            A = SparseTensor(row=row, col=col, value=edge_weight,
                             sparse_sizes=(N, N))
            S = SparseTensor(row=row, col=col, value=score, sparse_sizes=(N, N))
            S = S[:, perm]

            A = S.t() @ A @ S

            if self.add_self_loops:
                A = A.fill_diag(1.)
            else:
                A = A.remove_diag()

            row, col, edge_weight = A.coo()
            edge_index = torch.stack([row, col], dim=0)

            return x, edge_index, edge_weight, batch, perm

        else:
            mask = torch.zeros_like(fitness)
            mask[perm] = 1

            x1 = x * fitness.view(-1, 1) #x:[node_num, feature_dim] mask:[node_num]
            x2 = x1 * mask.view(-1, 1) #for these unselected nodes, set features to zero

            new_edge_weights = mask[edge_index[0]]+mask[edge_index[1]]
            edges_mask = torch.where(new_edge_weights != 2, torch.zeros(1, device=edge_index.device), torch.ones(1, device=edge_index.device))
            edge_weight2 = edge_weight * edges_mask

            return x2, edge_index, edge_weight2, batch, perm
class SAGPool_mix(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5, gnn_type='gcn', min_score=None,
                 multiplier=1, nonlinearity=torch.tanh, **kwargs):
        super(SAGPool_mix, self).__init__()
        self.gnn_type = gnn_type
        self.in_channels = in_channels
        self.ratio = ratio
        if gnn_type == 'gcn':
            self.gnn = GCNConv(in_channels, 1)
        elif gnn_type == 'mlp':
            self.gnn = Sequential(
                Linear(in_channels, int(in_channels//2)),
                ELU(),
                Linear(int(in_channels//2), 1))
        else:
            self.gnn = GraphConv(in_channels, 1)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gnn)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, attn=None, add_self_loop=False, remove_self_loop=False, ft=False):
        """"""
        if add_self_loop:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value=1)

        if edge_weight == None:
            edge_weight = torch.ones(edge_index.size()[1], device=x.device)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        if self.gnn_type == 'mlp':
            score = self.gnn(attn).view(-1)
        else:
            score = self.gnn(attn, edge_index, edge_weight=edge_weight).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        if ft:
            x = x[perm] * score[perm].view(-1, 1)
            x = self.multiplier * x if self.multiplier != 1 else x
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                               num_nodes=score.size(0))
            return x, edge_index, edge_weight, batch, perm
        else:
            mask = torch.zeros_like(score)
            mask[perm] = 1
            x1 = x * score.view(-1, 1)
            x2 = x1 * mask.view(-1, 1) #for these unselected nodes, set features to zero

            #edge_weights
            new_edge_weights = mask[edge_index[0]] + mask[edge_index[1]]
            edges_mask = torch.where(new_edge_weights != 2, torch.zeros(1, device=edge_index.device), torch.ones(1, device=edge_index.device))
            edge_weight2 = edge_weight * edges_mask


            return x2, edge_index, edge_weight2, batch, perm

class TOPKpooling_mix(TopKPooling):
    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, attn=None, add_self_loop=False, remove_self_loop=False, ft=False):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if add_self_loop:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value=1)

        if edge_weight == None:
            edge_weight = torch.ones(edge_index.size()[1], device=x.device)

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        if ft:
            x = x[perm] * score[perm].view(-1, 1)
            x = self.multiplier * x if self.multiplier != 1 else x

            batch = batch[perm]
            edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                               num_nodes=score.size(0))

            return x, edge_index, edge_weight, batch, perm
        else:
            mask = torch.zeros_like(score)
            mask[perm] = 1
            x1 = x * score.view(-1, 1)
            x2 = x1 * mask.view(-1, 1) #for these unselected nodes, set features to zero

            #edge_weights
            new_edge_weights = mask[edge_index[0]]+mask[edge_index[1]]
            edges_mask = torch.where(new_edge_weights != 2, torch.zeros(1, device=edge_index.device), torch.ones(1, device=edge_index.device))
            edge_weight2 = edge_weight * edges_mask

            return x2, edge_index, edge_weight2, batch, perm
class Hoppooling_mix(torch.nn.Module):
    def __init__(self, in_channels, ratio, walk_length=3):
        super(Hoppooling_mix, self).__init__()
        self.walk_length = walk_length
        self.pooling_ratio = ratio
    def reset_parameters(self):
        pass
    def forward(self, x, edge_index, batch=None, edge_weight=None, edge_attr=None,ft=False):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight == None:
            edge_weight = torch.ones(edge_index.size()[1], device=x.device)
        k_hops=[]
        num_nodes_1hop = scatter_add(edge_weight, edge_index[0], dim=0)
        k_hops.append(num_nodes_1hop)
        for i in range(int(self.walk_length) - 1):
            num_nodes_1hop = scatter_add(num_nodes_1hop[edge_index[1]] * edge_weight, edge_index[0], dim=0)
            k_hops.append(num_nodes_1hop)

        # x = scatter_add(x[edge_index[1]] * edge_weight, edge_index[0], dim=0)
        # score = num_nodes_1hop
        score = sum(k_hops)
        perm = topk(score, self.pooling_ratio, batch)
        if perm == None:
            print(score)
        if ft:
            x = x[perm]
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                               num_nodes=score.size(0))

            return x, edge_index, edge_weight, batch, perm

        else:
            mask = torch.zeros_like(score)
            mask[perm] = 1
            x2 = x * mask.view(-1, 1)  # for these unselected nodes, set features to zero

            # edge_weights
            new_edge_weights = mask[edge_index[0]] + mask[edge_index[1]]
            edges_mask = torch.where(new_edge_weights != 2, torch.zeros(1, device=edge_index.device),
                                     torch.ones(1, device=edge_index.device))
            edge_weight2 = edge_weight * edges_mask

            return x2, edge_index, edge_weight2, batch, perm
class Gappool_Mixed(MessagePassing):
    def __init__(self, in_channels, ratio):
        super(Gappool_Mixed, self).__init__()
        self.pooling_ratio = ratio
        self.linear = torch.nn.Linear(in_channels, 1)
    def reset_parameters(self):
        self.linear.reset_parameters()
    def forward(self, x, edge_index, batch=None, edge_weight=None, edge_attr=None, ft=False):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if edge_weight == None:
            edge_weight = torch.ones(edge_index.size()[1], device=x.device)
        # score = W \times sigma_{(x_u - x_j)^2} \times 0.5
        gap = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        gap = self.linear(gap) / 2
        score = gap.reshape(gap.size(0))
        perm = topk(score, self.pooling_ratio, batch)
        if ft:
            x = x[perm] * score[perm].view(-1, 1)
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                               num_nodes=score.size(0))

            return x, edge_index, edge_weight, batch, perm
        else:
            mask = torch.zeros_like(score)
            mask[perm] = 1
            x1 = x * score.view(-1, 1)
            x2 = x1 * mask.view(-1, 1)  # for these unselected nodes, set features to zero

            # edge_weights
            new_edge_weights = mask[edge_index[0]] + mask[edge_index[1]]
            edges_mask = torch.where(new_edge_weights != 2, torch.zeros(1, device=edge_index.device),
                                     torch.ones(1, device=edge_index.device))
            edge_weight2 = edge_weight * edges_mask
            return x2, edge_index, edge_weight2, batch, perm

    def message(self, x_i, x_j, edge_weight):
        out = (x_i - x_j)*(x_i - x_j)
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

def filter_features(x,edge_index, edge_weight, batch, th=0.001):
    score = torch.norm(x, p=1, dim=1)
    perm = topk(score, 0, batch, min_score=th)
    x = x[perm]
    batch = batch[perm]
    edge_index, edge_weight = filter_adj(edge_index, edge_weight, perm,
                                       num_nodes=score.size(0))
    # print('pool after mixed pooling: {},{}', x.size(0), edge_index.size(1))
    return x, edge_index, edge_weight, batch, perm

def filter_perm(x,edge_index, edge_weight, batch, perm_ori, th=0.001):
    perm = topk(perm_ori, 0, batch, min_score=th)
    x = x[perm]
    batch = batch[perm]
    edge_index, edge_weight = filter_adj(edge_index, edge_weight, perm,
                                       num_nodes=perm_ori.size(0))
    # print('pool after mixed pooling: {},{}', x.size(0), edge_index.size(1))
    return x, edge_index, edge_weight, batch, perm
