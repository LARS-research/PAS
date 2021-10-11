import torch
import torch.nn as nn
# from operations import *
from op_graph_classification import *
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.utils import add_self_loops,remove_self_loops,remove_isolated_nodes
def act_map(act):
  if act == "linear":
    return lambda x: x
  elif act == "elu":
    return torch.nn.functional.elu
  elif act == "sigmoid":
    return torch.sigmoid
  elif act == "tanh":
    return torch.tanh
  elif act == "relu":
    return torch.nn.functional.relu
  elif act == "relu6":
    return torch.nn.functional.relu6
  elif act == "softplus":
    return torch.nn.functional.softplus
  elif act == "leaky_relu":
    return torch.nn.functional.leaky_relu
  else:
    raise Exception("wrong activate function")


class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False, with_act=True):
    super(NaOp, self).__init__()

    self._op = NA_OPS[primitive](in_dim, out_dim)
    self.op_linear = nn.Linear(in_dim, out_dim)
    if not with_act:
      act = 'linear'
    self.act = act_map(act)
    self.with_linear = with_linear

  def reset_params(self):
    self._op.reset_params()
    self.op_linear.reset_parameters()

  def forward(self, x, edge_index, edge_weights):
    if self.with_linear:
      return self.act(self._op(x, edge_index, edge_weight=edge_weights) + self.op_linear(x))
    else:
      return self.act(self._op(x, edge_index, edge_weight=edge_weights))



class LaOp(nn.Module):
  def __init__(self, primitive, hidden_size, act, num_layers=None):
    super(LaOp, self).__init__()
    self._op = LA_OPS[primitive](hidden_size, num_layers)
    self.act = act_map(act)

  def reset_params(self):
    self._op.reset_params()

  def forward(self, x):
    # return self.act(self._op(x))
    return self._op(x)


class PoolingOp(nn.Module):
  def __init__(self, primitive, hidden, ratio, num_nodes=0):
    super(PoolingOp, self).__init__()
    self._op = POOL_OPS[primitive](hidden, ratio, num_nodes)
    self.primitive = primitive
  def reset_params(self):
    self._op.reset_params()
  def forward(self, x, edge_index,edge_weights, data, batch, mask):
    new_x, new_edge_index, _, new_batch, _ = self._op(x, edge_index, edge_weights, data, batch, mask, ft=True)
    return new_x, new_edge_index, new_batch, None


class ReadoutOp(nn.Module):
  def __init__(self, primitive, hidden):
    super(ReadoutOp, self).__init__()
    self._op = READOUT_OPS[primitive](hidden)
  def reset_params(self):
    self._op.reset_params()
  def forward(self, x, batch, mask):

    return self._op(x, batch, mask)



class NetworkGNN(nn.Module):

  def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='elu', args=None,is_mlp=False, num_nodes=0):
    super(NetworkGNN, self).__init__()
    self.genotype = genotype
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.in_dropout = in_dropout
    self.out_dropout = out_dropout
    self._criterion = criterion
    ops = genotype.split('||')
    self.args = args
    self.pooling_ratios = [[0.1], [0.25, 0.25], [0.5, 0.5, 0.5],[0.6, 0.6, 0.6, 0.6],[0.7, 0.7, 0.7, 0.7, 0.7],[0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]
    if self.args.data in ['NCI1', 'NCI109']:
      self.pooling_ratios = [[0.1], [0.5, 0.5], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7],
                             [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]

    self.bn = BatchNorm1d(hidden_size)
    self.pooling_ratio = self.pooling_ratios[num_layers-1]

    #node aggregator op
    self.lin1 = nn.Linear(in_dim, hidden_size)
    if self.args.search_act:
      act = ops[num_layers: num_layers*2]
    else:
      act = [act for i in range(num_layers)]
    self.gnn_layers = nn.ModuleList(
      [NaOp(ops[i], hidden_size, hidden_size, act[i], with_linear=args.with_linear) for i in range(num_layers)])

    if self.args.remove_pooling:
      poolops = ['none' for i in range(num_layers)]
    else:
      poolops = [ops[num_layers*2+i] for i in range(num_layers)]

    self.pooling_layers = nn.ModuleList(
      [PoolingOp(poolops[i], hidden_size, self.pooling_ratio[i]) for i in range(num_layers)])

    self.readout_layers = nn.ModuleList(
      [ReadoutOp(ops[num_layers*3 + i], hidden_size) for i in range(num_layers+1)])

    #learnable_LN
    if self.args.with_layernorm_learnable:
      self.lns_learnable = torch.nn.ModuleList()
      for i in range(self.num_layers):
        self.lns_learnable.append(torch.nn.BatchNorm1d(hidden_size))

    self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers+1)

    self.lin_output = nn.Linear(hidden_size, hidden_size)
    self.classifier = nn.Linear(hidden_size, out_dim)


  def reset_params(self):

    self.lin1.reset_parameters()

    for i in range(self.num_layers):
      self.gnn_layers[i].reset_params()
      self.pooling_layers[i].reset_params()

    for i in range(self.num_layers+1):
        self.readout_layers[i].reset_params()

    self.layer6.reset_params()
    self.lin_output.reset_parameters()
    self.classifier.reset_parameters()

  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    graph_representations = []

    x = F.elu(self.lin1(x))
    graph_representations.append(self.readout_layers[0](x, batch, None))

    x = F.dropout(x, p=self.in_dropout, training=self.training)
    # edge_weights = torch.ones(edge_index.size()[1], device=edge_index.device).float()
    for i in range(self.num_layers):
      x = self.gnn_layers[i](x, edge_index, edge_weights=None)
      # print('evaluate data {}-th gnn:'.format(i), x.size(), batch.size())
      if self.args.with_layernorm_learnable:
        x = self.lns_learnable[i](x)
      elif self.args.with_layernorm:
        layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
        x = layer_norm(x)
      if self.args.BN:
        x = self.bn(x)
      x = F.dropout(x, p=self.in_dropout, training=self.training)

      x, edge_index, batch, _ = self.pooling_layers[i](x, edge_index, None, data, batch, None)

      graph_representations.append(self.readout_layers[i+1](x, batch, None))

    x5 = self.layer6(graph_representations)

    x5 = F.elu(self.lin_output(x5))
    x5 = F.dropout(x5, p=self.out_dropout, training=self.training)

    logits = self.classifier(x5)
    return F.log_softmax(logits, dim=-1)

  def _loss(self, logits, target):
    return self._criterion(logits, target)



  def arch_parameters(self):
    return self._arch_parameters







