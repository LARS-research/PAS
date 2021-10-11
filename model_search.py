import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_self_loops,remove_self_loops
# from operations import *
from op_graph_classification import *
from torch.autograd import Variable
from genotypes import NA_PRIMITIVES, LA_PRIMITIVES, POOL_PRIMITIVES, READOUT_PRIMITIVES, ACT_PRIMITIVES
# from genotypes import Genotype
from torch_geometric.nn import  global_mean_pool,global_add_pool
from pooling_zoo import filter_features, filter_perm
def act_map(act):
    if act == "linear":
        return lambda x: x
    if act == "elu":
        return torch.nn.ELU
    elif act == "sigmoid":
        return torch.nn.Sigmoid
    elif act == "tanh":
        return torch.nn.Tanh
    elif act == "relu":
        return torch.nn.ReLU
    elif act == "relu6":
        return torch.nn.ReLU6
    elif act == "softplus":
        return torch.nn.Softplus
    elif act == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise Exception("wrong activate function")

class NaMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim, with_linear):
    super(NaMixedOp, self).__init__()
    self._ops = nn.ModuleList()

    for primitive in NA_PRIMITIVES:
      op = NA_OPS[primitive](in_dim, out_dim)
      self._ops.append(op)

      if with_linear:
        self._ops_linear = nn.ModuleList()
        op_linear = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear.append(op_linear)

      # self.act = act_map(act)

  def forward(self, x, weights, edge_index, edge_weights, with_linear):
    mixed_res = []
    if with_linear:
      for w, op, linear in zip(weights, self._ops, self._ops_linear):
        mixed_res.append(w * (op(x, edge_index, edge_weight=edge_weights)+linear(x)))
        # print('with linear')
    else:
      for w, op in zip(weights, self._ops):
        mixed_res.append(w * (op(x, edge_index, edge_weight=edge_weights)))
        # print('without linear')
    return sum(mixed_res)



class LaMixedOp(nn.Module):

  def __init__(self, hidden_size, num_layers=None):
    super(LaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in LA_PRIMITIVES:
      op = LA_OPS[primitive](hidden_size, num_layers)
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      # mixed_res.append(w * F.relu(op(x)))
      mixed_res.append(w * F.elu(op(x)))
    return sum(mixed_res)
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.float64, device=index.device)
    new_index = index.fill_(index[0]).type(torch.long)
    mask[new_index] = 1.0
    return mask
class PoolingMixedOp(nn.Module):
    def __init__(self, hidden, ratio, num_nodes=0):
        super(PoolingMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in POOL_PRIMITIVES:
            op = POOL_OPS[primitive](hidden, ratio, num_nodes)
            self._ops.append(op)

    def forward(self, x, edge_index, edge_weights, data, batch, mask, weights):
        new_x = []
        new_edge_weight = []
        new_perm = []
        # neither add or ewmove self_loop, so edge_index remain unchanged.
        for w, op in zip(weights, self._ops):
            # mixed_res.append(w * F.relu(op(x)))
            x_tmp, edge_index, edge_weight_tmp, batch, perm = op(x, edge_index, edge_weights, data, batch, mask)
            #print(perm.size(), w)
            new_x.append(x_tmp * w)
            new_edge_weight.append(w * edge_weight_tmp)
            new_perm.append(w * index_to_mask(perm, x.size(0)))


        #remove nodes with perm
        x, edge_index, edge_weight, batch, perm = filter_perm(sum(new_x), edge_index, sum(new_edge_weight), batch, sum(new_perm), th=0.01)
        return x, edge_index, edge_weight, batch, perm


class ReadoutMixedOp(nn.Module):
    def __init__(self, hidden):
        super(ReadoutMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in READOUT_PRIMITIVES:
            op = READOUT_OPS[primitive](hidden)
            self._ops.append(op)

    def forward(self, x, batch, mask, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            tmp_res = w * op(x, batch, mask)
            # print('readout', tmp_res.size())
            mixed_res.append(tmp_res)
        return sum(mixed_res)

class ActMixedOp(nn.Module):
    def __init__(self):
        super(ActMixedOp, self).__init__()
        self._ops = nn.ModuleDict()
        for primitive in ACT_PRIMITIVES:
            if primitive == 'linear':
                self._ops[primitive] = act_map(primitive)
            else:
                self._ops[primitive] = act_map(primitive)()

    def forward(self, x,  weights):
        mixed_res = []

        for i in range(len(ACT_PRIMITIVES)):
            mixed_res.append(weights[i] * self._ops[ACT_PRIMITIVES[i]](x))
        return sum(mixed_res)

class Network(nn.Module):

  def __init__(self, criterion, in_dim, out_dim, hidden_size, num_layers=3, dropout=0.5, epsilon=0.0, args=None, with_conv_linear=False,num_nodes=0 ):
    super(Network, self).__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_nodes = num_nodes
    self._criterion = criterion
    self.dropout = dropout
    self.epsilon = epsilon
    self.with_linear = with_conv_linear
    self.explore_num = 0
    self.args = args
    self.temp = args.temp
    self._loc_mean = args.loc_mean
    self._loc_std = args.loc_std


    if num_layers == 1:
        self.pooling_ratio = [0.1]
    elif num_layers == 2:
        self.pooling_ratio = [0.25, 0.25]
    elif num_layers == 3:
        self.pooling_ratio = [0.5, 0.5, 0.5]
    elif num_layers == 4:
        self.pooling_ratio = [0.6, 0.6, 0.6, 0.6]
    elif num_layers == 5:
        self.pooling_ratio = [0.7, 0.7, 0.7, 0.7, 0.7]
    elif num_layers == 6:
        self.pooling_ratio = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    self.lin1 = nn.Linear(in_dim, hidden_size)

    # node aggregator op
    self.gnn_layers = nn.ModuleList()
    for i in range(num_layers):
        self.gnn_layers.append(NaMixedOp(hidden_size, hidden_size, self.with_linear))

    #act op
    self.act_ops = nn.ModuleList()
    for i in range(num_layers):
        self.act_ops.append(ActMixedOp())

    #readoutop
    self.readout_layers = nn.ModuleList()
    for i in range(num_layers+1):
        self.readout_layers.append(ReadoutMixedOp(hidden_size))

    #pooling ops
    self.pooling_layers = nn.ModuleList()
    for i in range(num_layers):
        self.pooling_layers.append(PoolingMixedOp(hidden_size, self.pooling_ratio[i], num_nodes=self.num_nodes))

    #graph representation aggregator op
    self.layer6 = LaMixedOp(hidden_size, num_layers+1)

    self.lin_output = nn.Linear(hidden_size, hidden_size)
    self.classifier = nn.Linear(hidden_size, out_dim)

    self._initialize_alphas()


  def _get_categ_mask(self, alpha):
      # log_alpha = torch.log(alpha)
      log_alpha = alpha
      u = torch.zeros_like(log_alpha).uniform_()
      softmax = torch.nn.Softmax(-1)
      one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.temp)
      return one_hot

  def get_one_hot_alpha(self, alpha):
      one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
      idx = torch.argmax(alpha, dim=-1)

      for i in range(one_hot_alpha.size(0)):
        one_hot_alpha[i, idx[i]] = 1.0

      return one_hot_alpha
  def forward(self, data, discrete=False, mode='none'):
    self.args.search_act = False
    with_linear = self.with_linear
    x, edge_index = data.x, data.edge_index
    batch = data.batch
    # edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size()[0])

    if self.args.model_type == 'darts':
        na_alphas = F.softmax(self.log_na_alphas, dim=-1)
        la_alphas = F.softmax(self.log_la_alphas, dim=-1)
        pool_alphas = F.softmax(self.log_pool_alphas, dim=-1)
        readout_alphas = F.softmax(self.log_readout_alphas, dim=-1)
        act_alphas = F.softmax(self.log_act_alphas, dim=-1)
        # print('DARTS: sampled arch in train w', self._sparse(na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas))
    else:

        na_alphas = self._get_categ_mask(self.log_na_alphas)
        # sc_alphas = self._get_categ_mask(self.log_sc_alphas)
        la_alphas = self._get_categ_mask(self.log_la_alphas)
        pool_alphas = self._get_categ_mask(self.log_pool_alphas)
        readout_alphas = self._get_categ_mask(self.log_readout_alphas)
        act_alphas = self._get_categ_mask(self.log_act_alphas)
        # print('alpha in train w:',self._arch_parameters)
        # print('sampled arch in train w', self._sparse(na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas))

    if mode == 'evaluate_single_path':
        na_alphas = self.get_one_hot_alpha(na_alphas)
        la_alphas = self.get_one_hot_alpha(la_alphas)
        pool_alphas = self.get_one_hot_alpha(pool_alphas)
        readout_alphas = self.get_one_hot_alpha(readout_alphas)
        act_alphas = self.get_one_hot_alpha(act_alphas)

    graph_representations = []
    x = F.elu(self.lin1(x))
    edge_weights = torch.ones(edge_index.size()[1], device=edge_index.device).float()


    graph_representations.append(self.readout_layers[0](x, batch, None, readout_alphas[0]))

    for i in range(self.num_layers):
        x = self.gnn_layers[i](x, na_alphas[i], edge_index, edge_weights, with_linear)
        #print('evaluate data {}-th gnn:'.format(i), x.size(), batch.size())

        if self.args.search_act:
            x = self.act_ops[i](x, act_alphas[i])
        else:
            x = F.elu(x)
        layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
        x = layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x, edge_index, edge_weights, batch, _ = self.pooling_layers[i](x, edge_index, edge_weights, data, batch, None, pool_alphas[i])
        graph_representations.append(self.readout_layers[i+1](x, batch, None, readout_alphas[i+1]))
    x5 = self.layer6(graph_representations, la_alphas[0])

    x5 = F.elu(self.lin_output(x5))
    x5 = F.dropout(x5, p=self.dropout, training=self.training)
    logits = self.classifier(x5)

    return F.log_softmax(logits, dim=-1), [na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas]



  def _initialize_alphas(self):
    num_na_ops = len(NA_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)
    num_pool_ops = len(POOL_PRIMITIVES)
    num_readout_ops = len(READOUT_PRIMITIVES)
    num_act_ops = len(ACT_PRIMITIVES)

    if self.args.model_type == 'darts':
        self.log_na_alphas = Variable(1e-3*torch.randn(self.num_layers, num_na_ops).cuda(), requires_grad=True)
        self.log_act_alphas = Variable(1e-3*torch.randn(self.num_layers, num_act_ops).cuda(), requires_grad=True)
        self.log_pool_alphas = Variable(1e-3*torch.randn(self.num_layers, num_pool_ops).cuda(), requires_grad=True)
        self.log_readout_alphas = Variable(1e-3*torch.randn(self.num_layers+1, num_readout_ops).cuda(), requires_grad=True)
        self.log_la_alphas = Variable(1e-3*torch.randn(1, num_la_ops).cuda(), requires_grad=True)
    else:
        self.log_na_alphas = Variable(
            torch.ones(self.num_layers, num_na_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
        self.log_act_alphas = Variable(
            torch.ones(self.num_layers, num_act_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)

        self.log_pool_alphas = Variable(
            torch.ones(self.num_layers, num_pool_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
        self.log_readout_alphas = Variable(
            torch.ones(self.num_layers + 1, num_readout_ops).normal_(self._loc_mean, self._loc_std).cuda(),
            requires_grad=True)

        self.log_la_alphas = Variable(torch.ones(1, num_la_ops).normal_(self._loc_mean, self._loc_std).cuda(),
                                      requires_grad=True)

    self._arch_parameters = [
      self.log_na_alphas,
      self.log_act_alphas,
      self.log_pool_alphas,
      self.log_readout_alphas,
      self.log_la_alphas
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def _sparse(self, na_weights, act_alphas, pool_alphas, readout_alphas, la_weights):
      gene = []
      na_indices = torch.argmax(na_weights, dim=-1)
      for k in na_indices:
          gene.append(NA_PRIMITIVES[k])
      #sc_indices = sc_weights.argmax(dim=-1)

      act_indices = torch.argmax(act_alphas,dim=-1)
      for k in act_indices:
          gene.append(ACT_PRIMITIVES[k])

      pooling_indices = torch.argmax(pool_alphas, dim=-1)
      for k in pooling_indices:
          gene.append(POOL_PRIMITIVES[k])
      #la_indices = la_weights.argmax(dim=-1)

      readout_indices = torch.argmax(readout_alphas,dim=-1)
      for k in readout_indices:
          gene.append(READOUT_PRIMITIVES[k])

      la_indices = torch.argmax(la_weights, dim=-1)
      for k in la_indices:
          gene.append(LA_PRIMITIVES[k])
      return '||'.join(gene)
  def genotype(self):



    gene = self._sparse(F.softmax(self.log_na_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_act_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_pool_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_readout_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_la_alphas, dim=-1).data.cpu())
    return gene
