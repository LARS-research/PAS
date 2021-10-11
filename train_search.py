import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model_search import Network
from dataset import load_data
from logging_util import init_logger
graph_classification_dataset=['DD','MUTAG','PROTEINS','NCI1','NCI109','IMDB-BINARY','REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K']
node_classification_dataset = ['Cora','CiteSeer', 'PubMed','Amazon_Computers','Coauthor_CS','Coauthor_Physics','Amazon_Photo',
                               'small_Reddit', 'small_arxiv','Reddit','ogbn-arxiv']
parser = argparse.ArgumentParser("pas-train-search")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0005, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.08, help='learning rate for arch encoding')
parser.add_argument('--arch_learning_rate_min', type=float, default=0.005, help='minimum learning rate for arch encoding')
# parser.add_argument('--cos_arch_lr', action='store_true', default=False, help='lr decay for learning rate')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--with_conv_linear', type=bool, default=False, help=' in NAMixOp with linear op')
parser.add_argument('--num_layers', type=int, default=3, help='num of layers of GNN method.')
parser.add_argument('--withoutjk', action='store_true', default=False, help='remove la aggregtor')
parser.add_argument('--alpha_mode', type=str, default='train_loss', help='how to update alpha', choices=['train_loss', 'valid_loss', 'valid_acc'])
parser.add_argument('--search_act', action='store_true', default=False, help='search act in supernet.')
parser.add_argument('--hidden_size',  type=int, default=64, help='default hidden_size in supernet')
parser.add_argument('--BN',  type=int, default=64, help='default hidden_size in supernet')
parser.add_argument('--num_sampled_archs',  type=int, default=5, help='sample archs from supernet')


###for ablation stuty
parser.add_argument('--remove_pooling', action='store_true', default=False, help='remove pooling block.')
parser.add_argument('--remove_readout', action='store_true', default=False, help='exp5, only search the last readout block.')
parser.add_argument('--remove_jk', action='store_true', default=False, help='remove ensemble block, Graph representation = Z3')

#in the stage of update theta.
parser.add_argument('--temp', type=float, default=0.2, help=' temperature in gumble softmax.')
parser.add_argument('--loc_mean', type=float, default=10.0, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--lamda', type=int, default=2, help='sample lamda architectures in calculate natural policy gradient.')
parser.add_argument('--adapt_delta', action='store_true', default=False, help='adaptive delta in update theta.')
parser.add_argument('--delta', type=float, default=1.0, help='a fixed delta in update theta.')
parser.add_argument('--w_update_epoch', type=int, default=1, help='epoches in update W')
parser.add_argument('--model_type', type=str, default='snas', help='how to update alpha', choices=['darts', 'snas'])

args = parser.parse_args()
args.graph_classification_dataset = graph_classification_dataset
args.node_classification_dataset = node_classification_dataset
torch.set_printoptions(precision=4)
def main():
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_filename = os.path.join(args.save, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)
    num_features = num_classes = 0

    if args.data in args.graph_classification_dataset:
        data, num_nodes = load_data(args.data, batch_size=args.batch_size, split_seed=args.seed)
        num_features = data[0].num_features
        if args.data == 'COLORS-3':
            num_classes = 11
        else:
            num_classes = data[0].num_classes

    hidden_size = args.hidden_size

    criterion = F.nll_loss

    # criterion = criterion.cuda()
    model = Network(criterion, num_features, num_classes, hidden_size, epsilon=args.epsilon,
                    args=args, with_conv_linear=args.with_conv_linear, num_layers=args.num_layers, num_nodes=num_nodes)
    print('with_conv_linear: ', args.with_conv_linear)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        # momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    arch_optimizer = torch.optim.Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        # betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay) #fix lr in arch_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_arch = torch.optim.lr_scheduler.CosineAnnealingLR(arch_optimizer, float(args.epochs), eta_min=args.arch_learning_rate_min)
    # scheduler_arch = torch.optim.lr_scheduler.ExponentialLR(arch_optimizer, 0.98)
    # architect = Architect(model, args) # arch_parameter.
    test_acc_with_time = []
    cur_t = 0
    for epoch in range(args.epochs):
        t1 = time.time()
        lr = scheduler.get_lr()[0]
        arch_lr = scheduler_arch.get_lr()[0]
        if epoch % 1 == 0:
            logging.info('epoch %d lr %e', epoch, lr)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

        train_acc, train_obj = train_graph(data, model, criterion, optimizer, arch_optimizer,lr,arch_lr)
        scheduler.step()

        # valid_acc, valid_obj = infer(data, model, criterion, test=False)
        # test_acc, test_obj = infer(data, model, criterion, test=True)
        valid_obj, valid_acc, test_obj, test_acc = infer_graph(data, model, criterion)
        s_valid_obj, s_valid_acc, s_test_obj, s_test_acc = infer_graph(data, model, criterion, mode='evaluate_single_path')
        scheduler_arch.step()

        if epoch % 1 == 0:
            logging.info('epoch=%s, train_acc=%f, train_loss=%f, valid_acc=%f, valid_loss=%f, test_acc=%f, test_loss=%f, explore_num=%s', epoch, train_acc, train_obj, valid_acc, valid_obj, test_acc, test_obj, model.explore_num)
            print('epoch={}, train_acc={:.04f}, train_loss={:.04f},valid_acc={:.04f}, valid_loss={:.04f},test_acc={:.04f},test_loss={:.04f}, explore_num={}'.
                  format(epoch, train_acc, train_obj, valid_acc, valid_obj, test_acc, test_obj, model.explore_num))

            logging.info('single path evaluate. epoch=%s, valid_acc=%f, valid_loss=%f, test_acc=%f, test_loss=%f, explore_num=%s', epoch, s_valid_acc, s_valid_obj, s_test_acc, s_test_obj, model.explore_num)
            print('single_path evaluation.  epoch={}, valid_acc={:.04f}, valid_loss={:.04f},test_acc={:.04f},test_loss={:.04f}, explore_num={}'.
                  format(epoch, s_valid_acc, s_valid_obj, s_test_acc, s_test_obj, model.explore_num))
        utils.save(model, os.path.join(args.save, 'weights.pt'))

    logging.shutdown()
    return genotype

def train_graph(data, model,  criterion, model_optimizer, arch_optimizer, lr,arch_lr):
    model.train()
    total_loss = 0
    accuracy = 0
    # data:[dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    train_iters = data[4].__len__()//args.w_update_epoch + 1
    print('train_iters:{},train_data_num:{}'.format(train_iters, range(train_iters * args.w_update_epoch)))
    from itertools import cycle
    zip_train_data = list(zip(range(train_iters * args.w_update_epoch), cycle(data[4])))
    zip_valid_data = list(zip(range(train_iters), cycle(data[5])))

    # for train_data, valid_data in zip_list:
    for iter in range(train_iters):
        for i in range(args.w_update_epoch):
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            train_data = zip_train_data[iter*args.w_update_epoch+i][1].to(device)

            output, sample_z = model(train_data)
            output = output.to(device)
            accuracy += output.max(1)[1].eq(train_data.y.view(-1)).sum().item()
            #error loss and resource loss
            if args.data =='COLORS-3':
                # error_loss = (output-train_data.y.long()).pow(2).mean()
                error_loss = criterion(output, train_data.y.long())

            else:
                error_loss = criterion(output, train_data.y.view(-1))
            print('loss:{:.08f}'.format(error_loss.item()))

            total_loss += error_loss.item()
            arch_optimizer.zero_grad()
            error_loss.backward(retain_graph=True)
            model_optimizer.step()
        model_optimizer.zero_grad()
        if args.alpha_mode =='train_loss':
            arch_optimizer.step()
        if args.alpha_mode =='valid_loss':
            valid_data = zip_valid_data[iter][1].to(device)
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()
            output, _ = model(valid_data)
            output = output.to(device)

            error_loss = criterion(output, valid_data.y.view(-1))
            error_loss.backward()
            arch_optimizer.step()
    return accuracy/len(data[4].dataset), total_loss / len(data[4].dataset)
def infer_graph(data_, model, criterion, mode='none'):

    model.eval()
    valid_loss, test_loss = 0, 0
    valid_acc, test_acc = 0, 0

    for val_data in data_[5]:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits, _ = model(val_data, mode=mode)
            logits = logits.to(device)
        target = val_data.y
        if args.data == 'COLORS-3':
            # loss = (logits-target.long()).pow(2).mean()
            loss = criterion(logits, target.long())
        else:
            loss = criterion(logits, target)
        valid_loss += loss.item()
        valid_acc += logits.max(1)[1].eq(target.view(-1)).sum().item()
    for test_data in data_[6]:
        test_data = test_data.to(device)
        with torch.no_grad():
            logits,_ = model(test_data, mode=mode)
            logits = logits.to(device)
        target = test_data.y
        if args.data == 'COLORS-3':
            loss = criterion(logits, target.long())
            # loss = (logits-target.long()).pow(2).mean()
        else:
            loss = criterion(logits, target)
        test_loss += loss.item()
        test_acc += logits.max(1)[1].eq(target.view(-1)).sum().item()
    return valid_loss/len(data_[5].dataset), valid_acc/len(data_[5].dataset), \
           test_loss/len(data_[6].dataset), test_acc/len(data_[6].dataset)


def run_by_seed():
    res = []
    for i in range(args.num_sampled_archs):
        print('searched {}-th for {}...'.format(i+1, args.data))
        args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
        seed = np.random.randint(0, 10000)
        args.seed = seed
        genotype = main()
        res.append('seed={},genotype={},saved_dir={}'.format(seed, genotype, args.save))
    filename = './exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.data, filename))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please specify the required config info!!!')
        sys.exit(0)
    run_by_seed()


