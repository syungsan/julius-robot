#!/usr/bin/env python

# coding: utf-8

import csv
import numpy as np
import chainer
from chainer.datasets import tuple_dataset

# Dataset base-class handling csv
class Dataset:
  def __init__(self, csvfile):
    self.csvfile = csvfile

  def convert_to_dataset(self):
    data = []
    target = []
    with open(self.csvfile, 'r') as f:
      reader = csv.reader(f, delimiter=',')
      for columns in reader:
        target.append(columns[0])
        data.append(columns[1:])
    # convert to numpy arrays
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int32)
    return data, target

# for training dataset (returns tuple_dataset)
class TrainingDataset(Dataset):
  def __init__(self, csvfile):
    Dataset.__init__(self, csvfile)

  def convert_to_dataset(self):
    data, target = Dataset.convert_to_dataset(self)
    # returns with a dataset format preferred by chainer
    return tuple_dataset.TupleDataset(data, target)

# for test dataset (when uses a prediction)
class TestDataset(Dataset):
  def __init__(self, csvfile):
    Dataset.__init__(self, csvfile)

  def convert_to_dataset(self):
    return Dataset.convert_to_dataset(self)

