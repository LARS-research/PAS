import os
# import os.path as osp
import sys
import time
# import glob
# import pickle
# import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
# import genotypes
import torch.utils
# import torch_geometric.transforms as T
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
# from torch import cat
from torch_geometric.data import DataLoader
# from torch.autograd import Variable
from model import NetworkGNN as Network
# from utils import gen_uniform_60_20_20_split, save_load_split
from dataset import load_data, load_k_fold
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit,PPI
# from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from logging_util import init_logger
import torch.nn.functional as F

def main(exp_args):
    global train_args
    train_args = exp_args

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    train_args.save = 'logs/tune-{}-{}'.format(train_args.data, tune_str)
    if not os.path.exists(train_args.save):
        os.mkdir(train_args.save)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)



    #np.random.seed(train_args.seed)
    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    # np.random.seed(train_args.seed)
    # torch.backends.cudnn.deterministic = True

    num_features = num_classes = 0
    # if train_args.data == 'Amazon_Computers':
    #     data = Amazon('../data/AmazonComputers', 'Computers')
    # elif train_args.data == 'Amazon_Photo':
    #     data = Amazon('../data/AmazonPhoto', 'Photo')
    # elif train_args.data == 'Coauthor_Physics':
    #     data = Coauthor('../data/CoauthorPhysics', 'Physics')
    #
    # elif train_args.data == 'Coauthor_CS':
    #     data = Coauthor('../data/CoauthorCS', 'CS')
    #
    # elif train_args.data == 'Cora_Full':
    #     dataset = CoraFull('../data/Cora_Full')
    # elif train_args.data == 'PubMed':
    #     data = Planetoid('../data/', 'PubMed')
    # elif train_args.data == 'Cora':
    #     data = Planetoid('../data/', 'Cora')
    # elif train_args.data == 'CiteSeer':
    #     data = Planetoid('../data/', 'CiteSeer')
    # elif train_args.data == 'PPI':
    #     train_dataset = PPI('../data/PPI', split='train')
    #     val_dataset = PPI('../data/PPI', split='val')
    #     test_dataset = PPI('../data/PPI', split='test')
    #     num_features = train_dataset.num_features
    #     num_classes = train_dataset.num_classes
    #     ppi_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #     ppi_val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    #     ppi_test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    #     print('load PPI done!')
    #     data = [ppi_train_loader, ppi_val_loader, ppi_test_loader]

    if train_args.data in train_args.graph_classification_dataset:
        data, num_nodes = load_data(train_args.data, batch_size=train_args.batch_size)
        num_features = data[0].num_features
        num_classes = data[0].num_classes
        if train_args.data == 'COLORS-3':
            num_classes = 11
    hidden_size = train_args.hidden_size


    genotype = train_args.arch
    # if train_args.data == 'PPI':
    #     criterion = nn.BCEWithLogitsLoss()
    #     criterion = criterion.cuda()
    # else:
    criterion = F.nll_loss

    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, in_dropout=train_args.in_dropout, out_dropout=train_args.out_dropout,
                    act=train_args.activation, args = exp_args,is_mlp = train_args.is_mlp, num_nodes=num_nodes)
    model = model.cuda()

    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)
    print('param size = %fMB', utils.count_parameters_in_MB(model))
    def get_optimizer():
        if train_args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                train_args.learning_rate,
                # momentum=train_args.momentum,
                weight_decay=train_args.weight_decay
            )
        elif train_args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                train_args.learning_rate,
                momentum=train_args.momentum,
                weight_decay=train_args.weight_decay
            )
        elif train_args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(
                model.parameters(),
                train_args.learning_rate,
                weight_decay=train_args.weight_decay
            )
        return optimizer
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))
    if train_args.ft_mode == '10fold' and train_args.data in train_args.graph_classification_dataset:
        valid_losses = []
        valid_accs = []
        test_accs = []

        folds = 10
        for fold, data in enumerate(load_k_fold(data[0], folds, train_args.batch_size)):

            model.reset_params()
            optimizer = get_optimizer()
            if train_args.cos_lr:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs), eta_min=train_args.lr_min)
            print('#####Fold={}, train/val/test:{},{},{}'.format(fold, len(data[4].dataset), len(data[5].dataset), len(data[5].dataset)))
            for epoch in range(train_args.epochs):
                train_acc, train_obj = train_graph(data, model, criterion, optimizer)
                if train_args.cos_lr:
                    scheduler.step()

                valid_acc, valid_obj = infer_graph(data, model, criterion)
                test_acc, test_obj = infer_graph(data, model, criterion, test=True)
                valid_accs.append(valid_acc)
                valid_losses.append(valid_obj)
                test_accs.append(test_acc)
                if epoch % 10 == 0:
                    logging.info('fold=%s,epoch=%s, lr=%s, train_obj=%s, train_acc=%f, valid_acc=%s', fold, epoch,
                                 scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate, train_obj, train_acc, valid_acc)
                    print('fold={},epoch={}, lr={}, train_obj={:.08f}, train_acc={:.04f}, valid_loss={:.08f},valid_acc={:.04f},test_acc={:.04f}'.format(
                       fold, epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                        train_obj, train_acc, valid_obj, valid_acc, test_acc))

                if train_args.show_info:
                    print('fold={},epoch={}, lr={}, train_obj={:.08f}, train_acc={:.04f}, valid_loss={:.08f},valid_acc={:.04f},test_acc={:.04f}'.format(
                       fold, epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                        train_obj, train_acc, valid_obj, valid_acc, test_acc))

                utils.save(model, os.path.join(train_args.save, 'weights.pt'))

        # valid_losses, valid_accs, test_accs = torch.tensor(valid_losses), torch.tensor(valid_accs), torch.tensor(test_accs)
        valid_losses = torch.tensor(valid_losses).view(10, train_args.epochs)
        valid_accs = torch.tensor(valid_accs).view(10, train_args.epochs)
        test_accs = torch.tensor(test_accs).view(10, train_args.epochs)

        # min valid loss
        # valid_losses, argmin = valid_losses.min(dim=-1)
        # test_accs = test_accs[torch.arange(10, dtype=torch.long), argmin]
        # valid_accs = valid_accs[torch.arange(10, dtype=torch.long), argmin]
        # print('test_accs:', test_accs)

        # max_valid_acc
        valid_accs, argmax = valid_accs.max(dim=-1)
        valid_losses = valid_losses[torch.arange(10, dtype=torch.long), argmax]
        test_accs = test_accs[torch.arange(10, dtype=torch.long), argmax]
        print('test_accs:', test_accs)

        return valid_accs.mean().item(), test_accs.mean().item(), test_accs.std().item(), train_args
    else: #811 split
        optimizer = get_optimizer()
        model.reset_params()
        min_valid_loss = float("inf")
        best_valid_acc = 0
        best_test_acc = 0

        if train_args.cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs), eta_min=0.0001)
        for epoch in range(train_args.epochs):
            train_acc, train_obj = train_graph(data, model, criterion, optimizer)

            if train_args.cos_lr:
                scheduler.step()

            valid_acc, valid_obj = infer_graph(data, model, criterion)
            test_acc, test_obj = infer_graph(data, model, criterion, test=True)
            if valid_obj < min_valid_loss:
                min_valid_loss = valid_obj
                best_valid_acc = valid_acc
                best_test_acc = test_acc
            if epoch % 10 == 0:
                logging.info('epoch=%s, lr=%s, train_obj=%s, train_acc=%f, valid_acc=%s',
                             epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                             train_obj, train_acc, valid_acc)
            if train_args.show_info:
                print('epoch={}, lr={}, train_obj={:.08f}, train_acc={:.04f}, valid_loss={:.08f},valid_acc={:.04f},test_acc={:.04f}'.format(
                        epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                        train_obj, train_acc, valid_obj, valid_acc, test_acc))

            utils.save(model, os.path.join(train_args.save, 'weights.pt'))

        return best_valid_acc, best_test_acc,  0, train_args


def train_graph(data, model,  criterion, model_optimizer):
    model.train()
    total_loss = 0
    accuracy = 0

    # data:[dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader]
    for train_data in data[4]:

        train_data = train_data.to(device)
        model_optimizer.zero_grad()

        output = model(train_data).to(device)
        accuracy += output.max(1)[1].eq(train_data.y.view(-1)).sum().item()

        #error loss and resource loss
        if train_args.data =='COLORS-3':
            error_loss = criterion(output, train_data.y.long())
        else:
            error_loss = criterion(output, train_data.y.view(-1))

        total_loss += error_loss.item()

        error_loss.backward(retain_graph=True)
        model_optimizer.step()
    return accuracy/len(data[4].dataset), total_loss / len(data[4].dataset)

def infer_graph(data_, model, criterion, test=False):
    model.eval()
    total_loss = 0
    accuracy = 0
    #for valid or test.
    if test:
        data = data_[6]
    else:
        data = data_[5]
    for val_data in data:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits = model(val_data).to(device)
        target = val_data.y
        if train_args.data =='COLORS-3':
            loss = criterion(logits, target.long())
        else:
            loss = criterion(logits, target)
        total_loss += loss.item()
        accuracy += logits.max(1)[1].eq(target.view(-1)).sum().item()
    return accuracy / len(data.dataset), total_loss/len(data.dataset)

if __name__ == '__main__':
  main()


