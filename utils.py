import os
import os.path as osp
import numpy as np
import torch
import shutil
import yaml
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch import cat
class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

class EVLocalAvg(object):

    def __init__(self, window=5, ev_freq=2, total_epochs=50):
        """ Keep track of the eigenvalues local average.
        Args:
            window (int): number of elements used to compute local average.
                Default: 5
            ev_freq (int): frequency used to compute eigenvalues. Default:
                every 2 epochs
            total_epochs (int): total number of epochs that DARTS runs.
                Default: 50
        """
        self.window = window
        self.ev_freq = ev_freq
        self.epochs = total_epochs

        self.stop_search = False
        self.stop_epoch = total_epochs - 1
        self.stop_genotype = None

        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

        # start and end index of the local average window
        self.la_start_idx = 0
        self.la_end_idx = self.window

    def reset(self):
        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

    def update(self, epoch, ev, genotype):
        """ Method to update the local average list.
        Args:
            epoch (int): current epoch
            ev (float): current dominant eigenvalue
            genotype (namedtuple): current genotype
        """
        self.ev.append(ev)
        self.genotypes.update({epoch: genotype})
        # set the stop_genotype to the current genotype in case the early stop
        # procedure decides not to early stop
        self.stop_genotype = genotype

        # since the local average computation starts after the dominant
        # eigenvalue in the first epoch is already computed we have to wait
        # at least until we have 3 eigenvalues in the list.
        if (len(self.ev) >= int(np.ceil(self.window/2))) and (epoch < self.epochs - 1):
            # start sliding the window as soon as the number of eigenvalues in
            # the list becomes equal to the window size
            if len(self.ev) < self.window:
                self.ev_local_avg.append(np.mean(self.ev))
            else:
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx: self.la_end_idx]))
                self.la_start_idx += 1
                self.la_end_idx += 1

            # keep track of the offset between the current epoch and the epoch
            # corresponding to the local average. NOTE: in the end the size of
            # self.ev and self.ev_local_avg should be equal
            self.la_epochs.update({epoch: int(epoch - int(self.ev_freq*np.floor(self.window/2)))})

        elif len(self.ev) < int(np.ceil(self.window/2)):
            self.la_epochs.update({epoch: -1})

        # since there is an offset between the current epoch and the local
        # average epoch, loop in the last epoch to compute the local average of
        # these number of elements: window, window - 1, window - 2, ..., ceil(window/2)
        elif epoch == self.epochs - 1:
            for i in range(int(np.ceil(self.window/2))):
                assert len(self.ev[self.la_start_idx: self.la_end_idx]) == self.window - i
                self.ev_local_avg.append(np.mean(self.ev[self.la_start_idx:self.la_end_idx + 1]))
                self.la_start_idx += 1

    def early_stop(self, epoch, factor=1.18, es_start_epoch=10, delta=4):
        """ Early stopping criterion
        Args:
            epoch (int): current epoch
            factor (float): threshold factor for the ration between the current
                and prefious eigenvalue. Default: 1.3
            es_start_epoch (int): until this epoch do not consider early
                stopping. Default: 20
            delta (int): factor influencing which previous local average we
                consider for early stopping. Default: 2
        """
        if int(self.la_epochs[epoch] - self.ev_freq*delta) >= es_start_epoch:
            # the current local average corresponds to 
            # epoch - int(self.ev_freq*np.floor(self.window/2))
            current_la = self.ev_local_avg[-1]
            # by default take the local average corresponding to epoch
            # delta*self.ev_freq
            previous_la = self.ev_local_avg[-1 - delta]

            self.stop_search = current_la / previous_la > factor
            if self.stop_search:
                self.stop_epoch = int(self.la_epochs[epoch] - self.ev_freq*delta)
                self.stop_genotype = self.genotypes[self.stop_epoch]

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy_origin(output, target, topk=(1,)):

  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def accuracy(output, target, topk=(1,)):

    f1_res = f1_score(output,target, average='micro')
    #cal_f1
    res=[]
    res.append(f1_res)
    res.append(f1_res)
    return res
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_checkpoint(state, is_best, save, epoch, task_id):
    filename = "checkpoint_{}_{}.pth.tar".format(task_id, epoch)
    filename = os.path.join(save, filename)

    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def load_checkpoint(model, optimizer, architect, save, la_tracker, epoch, task_id):
    filename = "checkpoint_{}_{}.pth.tar".format(task_id, epoch)
    filename = os.path.join(save, filename)

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    model.alphas_normal.data = checkpoint['alphas_normal']
    model.alphas_reduce.data = checkpoint['alphas_reduce']
    optimizer.load_state_dict(checkpoint['optimizer'])
    architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
    la_tracker.ev = checkpoint['ev']
    la_tracker.ev_local_avg = checkpoint['ev_local_avg']
    la_tracker.genotypes = checkpoint['genotypes']
    la_tracker.la_epochs = checkpoint['la_epochs']
    la_tracker.la_start_idx = checkpoint['la_start_idx']
    la_tracker.la_end_idx = checkpoint['la_end_idx']
    lr = checkpoint['lr']
    return lr

def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def gen_uniform_60_20_20_split(data):
    skf = StratifiedKFold(5, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return cat(idx[:3], 0), cat(idx[3:4], 0), cat(idx[4:], 0)

def save_load_split(data, raw_dir, run, gen_splits):
    prefix = gen_splits.__name__[4:-6]
    # path = osp.join(raw_dir, '..', '{}_{:03d}.pt'.format(prefix, run))
    #
    # if osp.exists(path):
    #     split = torch.load(path)
    # else:
    #     split = gen_splits(data)
    #     torch.save(split, path)
    split = gen_splits(data)
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)

    return data

def write_yaml_results_eval(args, results_file, result_to_log):
  setting = '_'.join([args.space, args.data])
  regularization = '_'.join(
      [str(args.search_dp), str(args.search_wd)]
  )
  results_file = os.path.join(args.save, results_file+'.yaml')

  try:
    with open(results_file, 'r') as f:
      result = yaml.load(f)
    if setting in result.keys():
      if regularization in result[setting].keys():
        if args.search_task_id in result[setting][regularization]:
          result[setting][regularization][args.search_task_id].append(result_to_log)
        else:
          result[setting][regularization].update({args.search_task_id:
                                                 [result_to_log]})
      else:
        result[setting].update({regularization: {args.search_task_id:
                                                 [result_to_log]}})
    else:
      result.update({setting: {regularization: {args.search_task_id:
                                                [result_to_log]}}})
    with open(results_file, 'w') as f:
      yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)
  except (AttributeError, FileNotFoundError) as e:
    result = {
        setting: {
            regularization: {
                args.search_task_id: [result_to_log]
            }
        }
    }
    with open(results_file, 'w') as f:
      yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)

def write_yaml_results(args, results_file, result_to_log):
  setting = '_'.join([args.space, args.data])
  regularization = '_'.join(
      [str(args.drop_path_prob), str(args.weight_decay)]
  )
  results_file = os.path.join(args.save, results_file+'.yaml')

  try:
    with open(results_file, 'r') as f:
      result = yaml.load(f)
    if setting in result.keys():
      if regularization in result[setting].keys():
        result[setting][regularization].update({args.task_id: result_to_log})
      else:
        result[setting].update({regularization: {args.task_id: result_to_log}})
    else:
      result.update({setting: {regularization: {args.task_id: result_to_log}}})
    with open(results_file, 'w') as f:
      yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)
  except (AttributeError, FileNotFoundError) as e:
    result = {
        setting: {
            regularization: {
                args.task_id: result_to_log
            }
        }
    }
    with open(results_file, 'w') as f:
      yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)

