import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Conv1d, ELU, PReLU

from torch_geometric.nn import SAGEConv, GATConv, JumpingKnowledge
from torch_geometric.nn import GCNConv, GINConv,GraphConv,LEConv,SGConv,DenseSAGEConv,DenseGCNConv,DenseGINConv,DenseGraphConv
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool,global_sort_pool,GlobalAttention,Set2Set
from torch_geometric.nn import SAGPooling,TopKPooling,EdgePooling,ASAPooling,dense_diff_pool
from geniepath import GeniePathLayer
from pooling_zoo import SAGPool_mix, ASAPooling_mix, TOPKpooling_mix, Hoppooling_mix, Gappool_Mixed
from agg_zoo import GAT_mix,SAGE_mix,Geolayer_mix, GIN_mix
from torch_geometric.nn.inits import reset
NA_OPS = {
    #SANE
  'sage': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sage'),
  'sage_sum': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sum'),
  'sage_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),
  'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
  'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
  'gin': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin'),
  'gat_sym': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat_sym'),
  'gat_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'linear'),
  'gat_cos': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cos'),
  'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'generalized_linear'),
  'geniepath': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'geniepath'),
  'mlp': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'mlp'),


  #graph classification:
  'graphconv_add': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_add'),
  'graphconv_mean': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_mean'),
  'graphconv_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_max'),
  'sgc': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sgc'),
  'leconv': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'leconv'),

}
POOL_OPS = {

  'hoppool_1': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_1',num_nodes=num_nodes),
  'hoppool_2': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_2',num_nodes=num_nodes),
  'hoppool_3': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_3',num_nodes=num_nodes),

  'mlppool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'mlppool', num_nodes=num_nodes),
  'topkpool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'topkpool', num_nodes=num_nodes),

  'gappool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'gappool', num_nodes=num_nodes),

  'asappool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'asappool', num_nodes=num_nodes),
  'sagpool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'sagpool', num_nodes=num_nodes),
  'sag_graphconv': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'graphconv', num_nodes=num_nodes),

  'none': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'none', num_nodes=num_nodes),
}
READOUT_OPS = {
    "global_mean": lambda hidden :Readout_func('mean', hidden),
    "global_sum": lambda hidden  :Readout_func('add', hidden),
    "global_max": lambda hidden  :Readout_func('max', hidden),
    "none":lambda hidden  :Readout_func('none', hidden),
    'global_att': lambda hidden  :Readout_func('att', hidden),
    'global_sort': lambda hidden  :Readout_func('sort',hidden),
    'set2set': lambda hidden  :Readout_func('set2set',hidden),
}


LA_OPS={
  'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
  'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
  'l_mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers),
  'l_sum': lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
  'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers)
  #min/max
}

class NaAggregator(nn.Module):

  def __init__(self, in_dim, out_dim, aggregator):
    super(NaAggregator, self).__init__()
    #aggregator, K = agg_str.split('_')
    self.aggregator = aggregator
    if 'sage' == aggregator:
      # self._op = SAGEConv(in_dim, out_dim, normalize=True)
      self._op = SAGE_mix(in_dim, out_dim)
    if 'gcn' == aggregator:
      self._op = GCNConv(in_dim, out_dim)
    if 'gat' == aggregator:
      heads = 2
      out_dim /= heads
      self._op = GAT_mix(in_dim, int(out_dim), heads=heads, dropout=0.5)
    if 'gin' == aggregator:
      nn1 = Sequential(Linear(in_dim, out_dim), ELU(), Linear(out_dim, out_dim))
      self._op = GIN_mix(nn1)
    if aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
      heads = 2
      out_dim /= heads
      self._op = Geolayer_mix(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
    if aggregator in ['sum', 'max']:
      self._op = Geolayer_mix(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
    if aggregator in ['geniepath']:
      self._op = GeniePathLayer(in_dim, out_dim)
    if aggregator =='sgc':
        self._op = SGConv(in_dim, out_dim)
    if 'graphconv'in aggregator:
      aggr = aggregator.split('_')[-1]
      self._op = GraphConv(in_dim, out_dim, aggr=aggr)
    if aggregator == 'leconv':
        self._op = LEConv(in_dim, out_dim)
    if aggregator =='mlp':
      self._op = Sequential(Linear(in_dim, out_dim), ELU(), Linear(out_dim, out_dim))
  def reset_params(self):
    if self.aggregator == 'mlp':
      reset(self._op)
    else:
      self._op.reset_parameters()

  def forward(self, x, edge_index, edge_weight=None):
    if self.aggregator == 'mlp':
      return self._op(x)
    else:
      return self._op(x, edge_index, edge_weight=edge_weight)


class LaAggregator(nn.Module):

  def __init__(self, mode, hidden_size, num_layers=3):
    super(LaAggregator, self).__init__()
    self.mode = mode
    if self.mode in ['lstm', 'max', 'cat']:
      self.jump = JumpingKnowledge(mode, channels=hidden_size, num_layers=num_layers)
    if mode == 'cat':
        self.lin = Linear(hidden_size * num_layers, hidden_size)
    else:
        self.lin = Linear(hidden_size, hidden_size)
  def reset_params(self):
    self.lin.reset_parameters()
  def forward(self, xs):
    if self.mode in ['lstm', 'max', 'cat']:
      return self.lin(F.elu(self.jump(xs)))
    elif self.mode =='sum':
      return self.lin(F.elu(torch.stack(xs, dim=-1).sum(dim=-1)))
    elif self.mode =='mean':
      return self.lin(F.elu(torch.stack(xs, dim=-1).mean(dim=-1)))

class Readout_func(nn.Module):
  def __init__(self, readout_op, hidden):

    super(Readout_func, self).__init__()
    self.readout_op = readout_op

    if readout_op == 'mean':
      self.readout = global_mean_pool

    elif readout_op == 'max':
      self.readout = global_max_pool

    elif readout_op == 'add':
      self.readout = global_add_pool

    elif readout_op == 'att':
      self.readout = GlobalAttention(Linear(hidden, 1))

    elif readout_op == 'set2set':
      processing_steps = 2
      self.readout = Set2Set(hidden, processing_steps=processing_steps)
      self.s2s_lin = Linear(hidden*processing_steps, hidden)


    elif readout_op == 'sort':
      self.readout = global_sort_pool
      self.k = 10
      self.sort_conv = Conv1d(hidden, hidden, 5)#kernel size 3, output size: hidden,
      self.sort_lin = Linear(hidden*(self.k-5 + 1), hidden)
    elif readout_op =='mema':
      self.readout = global_mean_pool
      self.lin = Linear(hidden*2, hidden)
    elif readout_op == 'none':
      self.readout = global_mean_pool
    # elif self.readout_op == 'mlp':

  def reset_params(self):
    if self.readout_op =='sort':
      self.sort_conv.reset_parameters()
      self.sort_lin.reset_parameters()
    if self.readout_op in ['set2set', 'att']:
      self.readout.reset_parameters()
    if self.readout_op =='set2set':
      self.s2s_lin.reset_parameters()
    if self.readout_op == 'mema':
      self.lin.reset_parameters()
  def forward(self, x, batch, mask):
    #sparse data
    if self.readout_op == 'none':
      x = self.readout(x, batch)
      return x.mul(0.)
      # return None
    elif self.readout_op == 'sort':
      x = self.readout(x, batch, self.k)
      x = x.view(len(x), self.k, -1).permute(0, 2, 1)
      x = F.elu(self.sort_conv(x))
      x = x.view(len(x), -1)
      x = self.sort_lin(x)
      return x
    elif self.readout_op == 'mema':
      x1 = global_mean_pool(x, batch)
      x2 = global_max_pool(x, batch)
      x = torch.cat([x1, x2], dim=-1)
      x = self.lin(x)
      return x
    else:
      x = self.readout(x, batch)
      if self.readout_op == 'set2set':
        x = self.s2s_lin(x)
      return x



class Pooling_func(nn.Module):
  def __init__(self, hidden, ratio, op, dropout=0.6, num_nodes=0):
    super(Pooling_func, self).__init__()
    self.op = op
    self.max_num_nodes = num_nodes
    if op =='sagpool':
      self._op = SAGPool_mix(hidden, ratio=ratio, gnn_type='gcn')
    elif op =='mlppool':
      self._op = SAGPool_mix(hidden, ratio=ratio, gnn_type='mlp')
    elif op =='graphconv':
      self._op = SAGPool_mix(hidden, ratio=ratio, gnn_type='graphconv')

    elif 'hop' in op:
      hop_num = int(op.split('_')[-1])
      self._op = Hoppooling_mix(hidden, ratio=ratio, walk_length=hop_num)
    elif op == 'gappool':
      self._op = Gappool_Mixed(hidden, ratio=ratio)
    elif op == 'topkpool':
      # self._op = TopKPooling(hidden, ratio)
      self._op = TOPKpooling_mix(hidden, ratio=ratio)

    elif op == 'asappool':
      # self._op = ASAPooling(hidden, ratio, dropout=dropout)
      self._op = ASAPooling_mix(hidden, ratio=ratio, dropout=dropout)
  def reset_params(self):
    if self.op != 'none':
      self._op.reset_parameters()

  def forward(self, x, edge_index, edge_weights, data, batch, mask, ft=False):
    if self.op == 'none':
      perm = torch.ones(x.size(0), dtype=torch.float64, device=x.get_device())
      return x, edge_index, edge_weights, batch, perm

    elif self.op in ['asappool', 'topkpool', 'sagpool', 'mlppool', 'hoppool_1', 'hoppool_2', 'hoppool_3', 'gappool', 'graphconv']:
      # print('operations:', self.op)
      x, edge_index, edge_weight, batch, perm = self._op(x=x, edge_index=edge_index, edge_weight=edge_weights, batch=batch, ft=ft)
      return x, edge_index, edge_weight, batch, perm

