#!/usr/bin/env python

# coding: utf-8

# Network definition by example_mnist

import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
  def __init__(self, n_in, n_units, n_out):
    super(MLP, self).__init__(
      # the size of the inputs to each layer will be inferred
      l1=L.Linear(n_in, n_units),  # n_in -> n_units
      l2=L.Linear(n_units, n_units),  # n_units -> n_units
      l3=L.Linear(n_units, n_out),  # n_units -> n_out
    )

  def __call__(self, x):
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    return self.l3(h2)

