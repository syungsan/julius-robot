#!/usr/bin/env python

# coding: utf-8

# okaeri-kanojo classifier

from __future__ import print_function
import sys
import os
import argparse
import csv
import math
import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import mlp
import dataset
import play_okaeri

N_IN  = 25 # in
N_OUT = 2  # out
DEFAULT_UNIT = 500
MODEL_FILE = './train.model'

def get_args(parser):
  parser.add_argument('--gpu', '-g', type=int, default=-1)
  parser.add_argument('--unit', '-u', type=int, default=DEFAULT_UNIT)
  parser.add_argument('--testfile', '-t', default='',
            help='Input test file')
  args = parser.parse_args()

  if args.testfile == '':
    print('no input file')
    sys.exit(1)

  print('# test-file: {}'.format(args.testfile))
  return args

def main():
  parser = argparse.ArgumentParser(description='okaeri kanojo classifier')
  args = get_args(parser)

  model = L.Classifier(mlp.MLP(N_IN, args.unit, N_OUT))
  if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU
  chainer.serializers.load_npz(MODEL_FILE, model)

  # Load the dataset
  test_data = dataset.TestDataset(args.testfile)
  test, label = test_data.convert_to_dataset()

  # Classification
  v = model.predictor(test)
  prediction = np.argmax(v.data)
  print("okaeri_kanojo: classification: label=", label[0], "predict=", prediction)

  if label[0] == prediction:
    print("okaeri_kanojo: classification: okaeried")
    play_okaeri.play()
    sys.exit(0)
  else:
    sys.exit(1)

if __name__ == '__main__':
  main()
