from train4tune import main
import argparse

parser = argparse.ArgumentParser("pas")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--data', type=str, default='DD', choices=['DD', 'PROTEINS', 'NCI109', 'IMDB-BINARY', 'COX2', 'IMDB-MULTI'])
args1 = parser.parse_args()



DD_params = {'activation': 'relu', 'hidden_size': 32, 'in_dropout': 0.2, 'learning_rate': 0.006878897133613912,
             'model': 'SANE', 'optimizer': 'adam', 'out_dropout': 0.1, 'weight_decay': 1.8447814774886348e-05,
             'rnd_num': 1, 'ft_mode': '10fold', 'data': 'DD',
             'graph_classification_dataset': ['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K'],
             'node_classification_dataset': ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo'],
             'epochs': 100, 'is_mlp': False, 'batch_size': 64, 'arch': 'gcn||gcn||leaky_relu||leaky_relu||mlppool||gappool||global_sum||none||set2set||l_lstm',
             'gpu': 5, 'num_layers': 2, 'seed': 2, 'grad_clip': 5, 'momentum': 0.9, 'cos_lr': True, 'lr_min': 0.0, 'BN': False, 'with_linear': True,
             'with_layernorm': False, 'with_layernorm_learnable': True, 'show_info': False, 'withoutjk': False, 'search_act': False,
             'one_pooling': False, 'remove_pooling': False, 'remove_jk': False, 'remove_readout': False, 'fixpooling': 'null', 'fixjk': False}

PROTEINS_params = {'activation': 'elu', 'hidden_size': 16, 'in_dropout': 0.0, 'learning_rate': 0.007662125400401121,
                   'model': 'SANE', 'optimizer': 'adam', 'out_dropout': 0.6, 'weight_decay': 5.979931414632729e-05,
                   'rnd_num': 1, 'ft_mode': '10fold', 'data': 'PROTEINS',
                   'graph_classification_dataset': ['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI'],
                   'node_classification_dataset': ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo'],
                   'epochs': 100, 'is_mlp': False, 'batch_size': 64, 'arch': 'mlp||gin||leaky_relu||softplus||hoppool_3||sagpool||global_sum||set2set||set2set||l_max',
                   'gpu': 6, 'num_layers': 2, 'seed': 2, 'grad_clip': 5, 'momentum': 0.9, 'cos_lr': True, 'lr_min': 0.0, 'BN': False, 'with_linear': True,
                   'with_layernorm': False, 'with_layernorm_learnable': True, 'show_info': False, 'withoutjk': False, 'search_act': False,
                   'one_pooling': False, 'remove_pooling': False, 'remove_jk': False, 'remove_readout': False, 'fixpooling': 'null', 'fixjk': False}

IMDBB_params = {'activation': 'relu', 'hidden_size': 128, 'in_dropout': 0, 'learning_rate': 0.0037532981689691056,
                      'model': 'SANE', 'optimizer': 'adagrad', 'out_dropout': 0, 'weight_decay': 4.647697507807884e-05,
                      'rnd_num': 1, 'ft_mode': '10fold', 'data': 'IMDB-BINARY',
                      'graph_classification_dataset': ['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3'],
                      'node_classification_dataset': ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo'],
                      'epochs': 100, 'is_mlp': False, 'batch_size': 64, 'arch': 'gin||gcn||sigmoid||elu||hoppool_3||mlppool||global_sort||global_mean||global_att||l_sum',
                      'gpu': 6, 'num_layers': 2, 'seed': 2, 'grad_clip': 5, 'momentum': 0.9, 'cos_lr': True, 'lr_min': 0.0, 'BN': False, 'with_linear': True,
                      'with_layernorm': False, 'with_layernorm_learnable': True, 'show_info': False, 'withoutjk': False, 'search_act': False,
                      'one_pooling': False, 'remove_pooling': False, 'remove_jk': False, 'remove_readout': False, 'fixpooling': 'null', 'fixjk': False}

IMDBM_params = {'activation': 'elu', 'hidden_size': 64, 'in_dropout': 0.0, 'learning_rate': 0.008736477470973399,
                'model': 'SANE', 'optimizer': 'adagrad', 'out_dropout': 0.0, 'weight_decay': 0.0004603406614944098,
                'rnd_num': 1, 'ft_mode': '10fold', 'data': 'IMDB-MULTI',
                'graph_classification_dataset': ['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI'],
                'node_classification_dataset': ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo'],
                'epochs': 100, 'is_mlp': False, 'batch_size': 64, 'arch': 'gat||relu6||sag_graphconv||global_sum||global_mean||l_lstm',
                'gpu': 7, 'num_layers': 1, 'seed': 2, 'grad_clip': 5, 'momentum': 0.9, 'cos_lr': True, 'lr_min': 0.0, 'BN': False, 'with_linear': True,
                'with_layernorm': False, 'with_layernorm_learnable': True, 'show_info': False, 'withoutjk': False, 'search_act': False,
                'one_pooling': False, 'remove_pooling': False, 'remove_jk': False, 'remove_readout': False, 'fixpooling': 'null', 'fixjk': False}

COX2_patams = {'activation': 'relu', 'hidden_size': 16, 'in_dropout': 0.0, 'learning_rate': 0.001921018166741771,
               'model': 'SANE', 'optimizer': 'adam', 'out_dropout': 0.2, 'weight_decay': 6.593806454834699e-05,
               'rnd_num': 1, 'ft_mode': '10fold', 'data': 'COX2',
               'graph_classification_dataset': ['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI'],
               'node_classification_dataset': ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo'],
               'epochs': 140, 'is_mlp': False, 'batch_size': 64, 'arch': 'gin||gat||relu||tanh||none||gappool||global_sum||none||global_att||l_lstm',
               'gpu': 7, 'num_layers': 2, 'seed': 2, 'grad_clip': 5, 'momentum': 0.9, 'cos_lr': False, 'BN': False, 'lr_min': 0.0, 'with_linear': False,
               'with_layernorm': True, 'with_layernorm_learnable': False,'show_info': False, 'withoutjk': False, 'search_act': False,
               'one_pooling': False, 'remove_pooling': False, 'remove_jk': False, 'remove_readout': False, 'fixpooling': 'null', 'fixjk': False}

NCI109_params = {'activation': 'relu', 'hidden_size': 32, 'in_dropout': 0.1, 'learning_rate': 0.005385412452868308,
   'model': 'SANE', 'optimizer': 'adam', 'out_dropout': 0.1, 'weight_decay': 1.2956857470096877e-05,
   'rnd_num': 1, 'ft_mode': '10fold', 'data': 'NCI109',
   'graph_classification_dataset': ['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K'],
   'node_classification_dataset': ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo'],
   'epochs': 100, 'is_mlp': False, 'batch_size': 128, 'arch': 'graphconv_add||gin||relu6||leaky_relu||hoppool_1||mlppool||global_sum||set2set||global_att||l_lstm',
   'gpu': 3, 'num_layers': 2, 'seed': 2, 'grad_clip': 5, 'momentum': 0.9, 'cos_lr': True, 'lr_min': 0.0, 'BN': False, 'with_linear': True,
   'with_layernorm': False, 'with_layernorm_learnable': True, 'show_info': False, 'withoutjk': False, 'search_act': False,
   'one_pooling': False, 'remove_pooling': False, 'remove_jk': False, 'remove_readout': False, 'fixpooling': 'null', 'fixjk': False}

params_dict = {
    'DD': DD_params,
    'PROTEINS': PROTEINS_params,
    'NCI109': NCI109_params,
    'COX2': COX2_patams,
    'IMDB-BINARY': IMDBB_params,
    'IMDB-MULTI': IMDBM_params
}

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

args = Dict()
params_dict[args1.data]['gpu'] = args1.gpu
for k, v in params_dict[args1.data].items():
    args[k] = v

for i in range(5):
    valid_acc, test_acc, test_std, args = main(args)
    print('{}/5: valid_acc:{:.04f}, test_acc: {:.04f}+-{:.04f}'.format(i, valid_acc, test_acc, test_std))


