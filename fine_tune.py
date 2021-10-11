import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
import random
from logging_util import init_logger
from train4tune import main
# from test4tune import main as test_main
import torch

sane_space ={'model': 'SANE',
         'hidden_size': hp.choice('hidden_size', [16, 32, 64, 128, 256, 512]),
         # 'hidden_size': hp.choice('hidden_size', [128, ]),
         'learning_rate': hp.uniform("lr", 0.005, 0.05),
         'weight_decay': hp.uniform("wr", -5, -3),
         'optimizer': hp.choice('opt', ['adam', 'adagrad']),
         'in_dropout': hp.randint('in_dropout', 10),
         'out_dropout': hp.randint('out_dropout', 10),
         'activation': hp.choice('act', ['relu', 'elu'])
         # 'activation': hp.choice('act', ["sigmoid", "tanh", "relu", "softplus", "leaky_relu", "relu6", "elu"])
         }
graph_classification_dataset=['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2',  'IMDB-MULTI','COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K']
node_classification_dataset = ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo']
def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in SANE')
    parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--with_layernorm_learnable', action='store_true', default=False, help='use the learnable layer norm')
    parser.add_argument('--BN',  action='store_true', default=False,  help='use BN.')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size of data.')
    parser.add_argument('--is_mlp', action='store_true', default=False, help='is_mlp')
    parser.add_argument('--ft_weight_decay', action='store_true', default=False, help='with weight decay in finetune stage.')
    parser.add_argument('--ft_dropout', action='store_true', default=False, help='with dropout in finetune stage')
    parser.add_argument('--ft_mode', type=str, default='10fold', choices=['811', '622', '10fold'], help='data split function.')
    parser.add_argument('--hyper_epoch', type=int, default=50, help='hyper epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs for each model')
    parser.add_argument('--cos_lr',  action='store_true', default=False,  help='use cos lr.')
    parser.add_argument('--lr_min',  type=float, default=0.0,  help='use cos lr.')
    parser.add_argument('--show_info',  action='store_true', default=False,  help='print training info in each epoch')
    parser.add_argument('--withoutjk', action='store_true', default=False, help='remove la aggregtor')
    parser.add_argument('--search_act', action='store_true', default=False, help='search act in supernet.')
    parser.add_argument('--one_pooling', action='store_true', default=False, help='only one pooling layers after 2th layer.')
    parser.add_argument('--seed', type=int, default=2, help='seed for finetune')
    parser.add_argument('--remove_pooling', action='store_true', default=False,
                        help='remove pooling block.')
    parser.add_argument('--remove_readout', action='store_true', default=False,
                        help='remove readout block. Only search the last readout block.')
    parser.add_argument('--remove_jk', action='store_true', default=False,
                        help='remove ensemble block. In the last readout block,use global sum. Graph representation = Z3')
    parser.add_argument('--fixpooling', type=str, default='null',
                        help='use fixed pooling functions')
    parser.add_argument('--fixjk',action='store_true', default=False,
                        help='use concat,rather than search from 3 ops.')

    global args1
    args1 = parser.parse_args()
    random.seed(args1.seed)
    torch.cuda.manual_seed_all(args1.seed)
    torch.manual_seed(args1.seed)
    np.random.seed(args1.seed)
    os.environ.setdefault("HYPEROPT_FMIN_SEED", str(args1.seed))
class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = ARGS()
    for k, v in arg_map.items():
        setattr(args, k, v)
    for k, v in args1.__dict__.items():
        setattr(args, k, v)
    setattr(args, 'rnd_num', 1)

    # args.ft_mode = args1.ft_mode
    if args1.ft_dropout:
        args.in_dropout = args.in_dropout / 10.0
        args.out_dropout = args.out_dropout / 10.0
    else:
        args.in_dropout = args.out_dropout = 0

    if args1.ft_weight_decay:
        args.weight_decay = 10**args.weight_decay
    else:
        args.weight_decay = 0

    args.graph_classification_dataset = graph_classification_dataset
    args.node_classification_dataset = node_classification_dataset

    args.grad_clip = 5
    args.momentum = 0.9

    return args

def objective(args):
    args = generate_args(args)

    vali_acc, test_acc, test_std, args = main(args)
    print(args)
    return {
        'loss': -vali_acc,
        'test_acc': test_acc,
        'test_std': test_std,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
        }

def run_fine_tune():

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = 'logs/tune-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
        os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    init_logger('fine-tune', log_filename, logging.INFO, False)

    lines = open(args1.arch_filename, 'r').readlines()

    suffix = args1.arch_filename.split('_')[-1][:-4]  # need to re-write the suffix?

    test_res = []
    arch_set = set()

    if args1.search_act:
        #search act in train supernet. Finetune stage remove act search.
        sane_space['activation'] = hp.choice("act", [0])

    if not args1.ft_weight_decay:
        sane_space['weight_decay'] = hp.choice("wr", [0])
    if not args1.ft_dropout:
        sane_space['in_dropout'] = hp.choice('in_dropout', [0])
        sane_space['out_dropout'] = hp.choice('out_dropout', [0])
    if args1.data in ['COLLAB', 'REDDIT-MULTI-5K', 'NCI1', 'NCI109']:
        sane_space['learning_rate'] = hp.uniform("lr", 0.005, 0.02)
    for ind, l in enumerate(lines):
        try:
            print('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), log_filename))
            logging.info('**********process {}-th/{}**************8'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]
            args1.arch = arch
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()

            # start = time.time()
            start = time.time()
            trials = Trials()
            #tune with validation acc, and report the test accuracy with the best validation acc
            best = fmin(objective, sane_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)), max_evals=args1.hyper_epoch, trials=trials)

            space = hyperopt.space_eval(sane_space, best)
            args = generate_args(space)
            print('best space is ', args.__dict__)
            res['tuned_args'] = args.__dict__

            record_time_res = []
            c_vali_acc, c_test_acc = 0, 0
            #report the test acc with the best vali acc
            for d in trials.results:
                if -d['loss'] > c_vali_acc:
                    c_vali_acc = -d['loss']
                    c_test_acc = d['test_acc']
                    record_time_res.append('%s,%s,%s' % (d['eval_time'] - start, c_vali_acc, c_test_acc))

            res['test_acc'] = c_test_acc
            print('test_acc={}'.format(c_test_acc))

            #cal std and record the best results.
            test_accs = []
            test_stds=[]
            for i in range(5):
                # args.epochs = 100
                vali_acc, t_acc, t_std, test_args = main(args)
                print('cal std: times:{}, valid_Acc:{}, test_acc:{:.04f}+-{:.04f}'.format(i, vali_acc, t_acc, t_std))
                test_accs.append(t_acc)
                test_stds.append(t_std)
            test_accs = np.array(test_accs)
            test_stds = np.array(test_stds)
            for i in range(5):
                print('Train from scratch {}/5: Test_acc:{:.04f}+-{:.04f}'.format(i, test_accs[i], test_stds[i]))
            print('test_results_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))
            res['accs_train_from_scratch'] = test_accs
            res['stds_train_from_scratch'] = test_stds
            test_res.append(res)

            with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('test_results of 5 times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))
            logging.info('**********finish {}-th/{}**************8'.format(ind + 1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind + 1, l.strip(), e)
            import traceback
            traceback.print_exc()
    print('finsh tunining {} archs, saved in {}'.format(len(arch_set),
                                                        'tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix)))


if __name__ == '__main__':
    get_args()

    if args1.arch_filename:
        run_fine_tune()


