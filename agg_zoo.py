import torch_geometric
import torch
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn import SAGEConv, GATConv, JumpingKnowledge
from torch_geometric.nn import GCNConv, GINConv,GraphConv,LEConv,SGConv,DenseSAGEConv,DenseGCNConv,DenseGINConv,DenseGraphConv
from pyg_gnn_layer import GeoLayer



class GAT_mix(GATConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, edge_weight: OptTensor = None):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None
        self.add_self_loops = False
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] if size is not None else num_nodes
                num_nodes = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size, edge_weight=edge_weight)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, edge_weight: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x1 = x_j * alpha.unsqueeze(-1)
        if edge_weight is None:
            return x1
        else:
            x2 = (x1.view(-1, self.heads * self.out_channels).t() * edge_weight).t().view(-1, self.heads, self.out_channels)
            return x2

class SAGE_mix(SAGEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class GIN_mix(GINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class Geolayer_mix(GeoLayer):
    def forward(self, x, edge_index, size=None, edge_weight: OptTensor = None):
        """"""
        # rm self loop
        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # prepare
        if torch.is_tensor(x):
            x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight).view(-1, self.heads, self.out_channels),
                 None if x[1] is None else torch.matmul(x[1], self.weight).view(-1, self.heads, self.out_channels))
        num_nodes = x.size(0) if torch.is_tensor(x) else size[0]
        return self.propagate(edge_index, size=size, x=x, num_nodes=num_nodes, edge_weight=edge_weight)
    def message(self, x_i, x_j, edge_index, num_nodes, edge_weight=None):
        if self.att_type == "const":
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            neighbor = x_j

        else:
            # Compute attention coefficients.
            alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
            alpha = softmax(alpha, edge_index[0], ptr=None, num_nodes=num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            neighbor = x_j * alpha.view(-1, self.heads, 1)
            # if self.pool_dim > 0:
            #    for layer in self.pool_layer:
            #        neighbor = layer(neighbor)
        if edge_weight is None:
            return neighbor
        else:
            x2 = (neighbor.view(-1, self.heads * self.out_channels).t() * edge_weight).t().view(-1, self.heads, self.out_channels)
            return x2








