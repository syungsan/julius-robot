#!/usr/bin/env python

# coding: utf-8

# tadaima training script

from __future__ import print_function
import argparse
import csv
import numpy

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import mlp
import dataset

N_IN  = 25 # in
N_OUT = 2  # out
DEFAULT_UNIT = 500
TRAIN_FILE = './train.csv'   # input training file
MODEL_FILE = './train.model' # output model file
# OPT_FILE   = './train.optimizer'

def get_args(parser):
  parser.add_argument('--batchsize', '-b', type=int, default=100)
  parser.add_argument('--epoch', '-e', type=int, default=20)
  parser.add_argument('--gpu', '-g', type=int, default=-1)
  parser.add_argument('--unit', '-u', type=int, default=DEFAULT_UNIT)
  parser.add_argument('--out', '-o', default='result',
            help='Directory to output the result')
  parser.add_argument('--resume', '-r', default='',
            help='Resume the training from snapshot')
  args = parser.parse_args()

  print('GPU: {}'.format(args.gpu))
  print('# unit: {}'.format(args.unit))
  print('# Minibatch-size: {}'.format(args.batchsize))
  print('# epoch: {}'.format(args.epoch))
  print('')

  return args

def main():
  parser = argparse.ArgumentParser(description='okaeri kanojo trainer')
  args = get_args(parser)

  model = L.Classifier(mlp.MLP(N_IN, args.unit, N_OUT))
  if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

  # Setup an optimizer
  optimizer = chainer.optimizers.Adam()
  optimizer.setup(model)

  # Load the sprecog dataset
  train_data = dataset.TrainingDataset(TRAIN_FILE)
  train = train_data.convert_to_dataset()
  test = train

  train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
  test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                         repeat=False, shuffle=False)

  # Set up a trainer
  updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
  trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

  # Evaluate the model with the test dataset for each epoch
  trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

  # Dump a computational graph from 'loss' variable at the first iteration
  # The "main" refers to the target link of the "main" optimizer.
  trainer.extend(extensions.dump_graph('main/loss'))

  # Take a snapshot at each epoch
  trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

  # Write a log of evaluation statistics for each epoch
  trainer.extend(extensions.LogReport())

  # Print selected entries of the log to stdout
  trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

  # Print a progress bar to stdout
  trainer.extend(extensions.ProgressBar())

  if args.resume:
    # Resume from a snapshot
    chainer.serializers.load_npz(args.resume, trainer)

  # Run the training
  trainer.run()

  # Save the model
  chainer.serializers.save_npz(MODEL_FILE, model)
  # chainer.serializers.save_npz(OPT_FILE, optimizer)

if __name__ == '__main__':
  main()
