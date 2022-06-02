import os
import collections
import logging
import glob
import re
import cv2 
import torch, torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import datasets, transforms as T
from sklearn.model_selection import train_test_split

from torch.utils.data.dataset import Dataset

import itertools as it
import copy

import random
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision import transforms 
from torchvision.transforms import Compose 
torch.backends.cudnn.benchmark=True


#-------------------------------------------------------------------------------------------------------
# DATASETS
#-------------------------------------------------------------------------------------------------------
DATA_PATH = r'C:\Users\gabri\Mestrado2\Project HIIAC\Stc'

classes_pc = 2
 #False: non_iid dataset, True: Real-world dataset

def get_kws():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data = np.load(os.path.join(DATA_PATH, "Speech_Commands/data_noaug.npz"))
  
  x_train, y_train = data["x_train"], data["y_train"].astype("int").flatten()
  x_test, y_test = data["x_test"], data["y_test"].astype("int").flatten()

  x_train, x_test = (x_train-0.5412386)/0.2746128, (x_test-0.5412386)/0.2746128
  
  return x_train, y_train, x_test, y_test

def get_kuhar():
    df = pd.read_csv("KU-HAR_time_domain_subsamples_20750x300.csv",header=None)
    dff = df.values
    X = dff[:, 0: 1800]
    X = np.array(X, dtype=np.float32)
    y = dff[:, 1800]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(df.tail())
    return X_train, y_train, X_test, y_test

    
    
def get_mnist():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=True, download=True) 
  data_test = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=False, download=True) 
  
  x_train, y_train = data_train.train_data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.train_labels)
  x_test, y_test = data_test.test_data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.test_labels)
  
  return x_train, y_train, x_test, y_test


def get_fashionmnist():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=True, download=True) 
  data_test = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=False, download=True) 
  
  x_train, y_train = data_train.train_data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.train_labels)
  x_test, y_test = data_test.test_data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.test_labels)

  return x_train, y_train, x_test, y_test


def get_cifar10():
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  transform = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor()])

  data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    
  data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

  x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
  x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("\nData: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
      np.min(labels_test), np.max(labels_test)))

def clients_rand(train_len, nclients):
  '''
  train_len: size of the train data
  nclients: number of clients
  
  Returns: to_ret
  
  This function creates a random distribution 
  for the clients, i.e. number of images each client 
  possess.
  '''
  client_tmp=[]
  sum_=0
  #### creating random values for each client ####
  for i in range(nclients-1):
    tmp=random.randint(10,100)
    sum_+=tmp
    client_tmp.append(tmp)

  client_tmp= np.array(client_tmp)
  #### using those random values as weights ####
  clients_dist= ((client_tmp/sum_)*train_len).astype(int)
  num  = train_len - clients_dist.sum()
  to_ret = list(clients_dist)
  to_ret.append(num)
  return to_ret

def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds any number of classes which is trying to simulate real world dataset
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels(10)
    n_clients : number of clients
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : splitted client data into desired format
  '''
  def break_into(n,m):
    ''' 
    return m random integers with sum equal to n 
    '''
    to_ret = [1 for i in range(m)]
    for i in range(n-m):
        ind = random.randint(0,m-1)
        to_ret[ind] += 1
    return to_ret

  #### constants ####
  n_classes = len(set(labels))
  classes = list(range(n_classes))
  np.random.shuffle(classes)
  label_indcs  = [list(np.where(labels==class_)[0]) for class_ in classes]
  
  #### classes for each client ####
  tmp = [np.random.randint(1,10) for i in range(n_clients)]
  total_partition = sum(tmp)

  #### create partition among classes to fulfill criteria for clients ####
  class_partition = break_into(total_partition, len(classes))

  #### applying greedy approach first come and first serve ####
  class_partition = sorted(class_partition,reverse=True)
  class_partition_split = {}

  #### based on class partition, partitioning the label indexes ###
  for ind, class_ in enumerate(classes):
      class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind],class_partition[ind])]
      
#   print([len(class_partition_split[key]) for key in  class_partition_split.keys()])

  clients_split = []
  count = 0
  for i in range(n_clients):
    n = tmp[i]
    j = 0
    indcs = []

    while n>0:
        class_ = classes[j]
        if len(class_partition_split[class_])>0:
            indcs.extend(class_partition_split[class_][-1])
            count+=len(class_partition_split[class_][-1])
            class_partition_split[class_].pop()
            n-=1
        j+=1

    ##### sorting classes based on the number of examples it has #####
    classes = sorted(classes,key=lambda x:len(class_partition_split[x]),reverse=True)
    if n>0:
        raise ValueError(" Unable to fulfill the criteria ")
    clients_split.append([data[indcs], labels[indcs]])
#   print(class_partition_split)
#   print("total example ",count)


  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
      
    if verbose:
      print_split(clients_split)
  
  clients_split = np.array(clients_split)
  
  return clients_split

#-------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS
#-------------------------------------------------------------------------------------------------------
def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):
  '''
  Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
  different labels
  data : [n_data x shape]
  labels : [n_data (x 1)] from 0 to n_labels
  '''
  # constants
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1
  
  if balancedness >= 1.0:
    data_per_client = [n_data // n_clients]*n_clients
    data_per_client_per_class = [data_per_client[0] // classes_per_client]*n_clients
  else:
    fracs = balancedness**np.linspace(0,n_clients-1, n_clients)
    fracs /= np.sum(fracs)
    fracs = 0.1/n_clients + (1-0.1)*fracs
    data_per_client = [np.floor(frac*n_data).astype('int') for frac in fracs]

    data_per_client = data_per_client[::-1]

    data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]

  if sum(data_per_client) > n_data:
    print("Impossible Split")
    exit()
  
  # sort for labels
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)
    
  # split data among clients
  clients_split = []
  c = 0
  for i in range(n_clients):
    client_idcs = []
    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
      
    clients_split += [(data[client_idcs], labels[client_idcs])]
  
  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
      
  if verbose:
    print_split(clients_split)
        
  return clients_split

def shuffle_list(data):
  '''
  This function returns the shuffled data
  '''
  for i in range(len(data)):
    #print(data[i][1])
    tmp_len= len(data[i][0])
    index = [i for i in range(tmp_len)]
    random.shuffle(index)
    #data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
  return data


def shuffle_list_data(x, y):
  '''
  This function is a helper function, shuffles an
  array while maintaining the mapping between x and y
  '''
  inds = list(range(len(x)))
  random.shuffle(inds)
  return x[inds],y[inds]

#-------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS
#-------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms 

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]
          

def get_default_data_transforms(name, train=True, verbose=True):
  transforms_train = {
  'mnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.06078,),(0.1957,))
    ]),
  'fashionmnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
  'kws' : None,
  'kuhar': None
  }
  transforms_eval = {
  'mnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.06078,),(0.1957,))
    ]),
  'fashionmnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#
  'kws' : None,
  'kuhar' : None
  }

  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train[name].transforms:
      print(' -', transformation)
    print()

  return (transforms_train[name], transforms_eval[name])


def get_data_loaders(hp, verbose=True):
  real_wd = True
  x_train, y_train, x_test, y_test = globals()['get_'+hp['dataset']]()

  if verbose:
    print_image_data_stats(x_train, y_train, x_test, y_test)

  transforms_train, transforms_eval = get_default_data_transforms(hp['dataset'], verbose=False)

  if real_wd:
    split = split_image_data_realwd(x_train, y_train, n_clients=hp['n_clients'], verbose = verbose)
  else:  
    split = split_image_data(x_train, y_train,n_clients=hp['n_clients'], 
          classes_per_client=hp['classes_per_client'], balancedness=hp['balancedness'], verbose=verbose)
  
  #print(split)
  split_tmp = shuffle_list(split)

  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
                                                                batch_size=hp['batch_size'], shuffle=True) for x, y in split]
  train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train, transforms_eval), batch_size=100, shuffle=False)
  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

  
  stats = {"split" : [x.shape[0] for x, y in split]}
  return client_loaders, train_loader, test_loader, stats

