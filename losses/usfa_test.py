"""Tests for utils."""

from absl.testing import absltest

import collections
import rlax
import haiku as hk
import chex
import jax
import jax.numpy as jnp
import numpy as np

chex.set_n_cpu_devices(4)

from losses import usfa

class UsfaAuxTest(absltest.TestCase):

  def test_qlearning(self):
    time_size = 4
    batch_size = 5
    num_samples = 11
    action_size = 3
    cumulants = 2

    preds = {
      'w_embed' : np.random.rand(time_size, batch_size, cumulants),
      'sf' : np.random.rand(time_size, batch_size, num_samples, action_size, cumulants)
    }
    preds = collections.namedtuple('preds', preds.keys())(**preds)

    data = {
      'reward' : np.zeros((time_size, batch_size)),
      'discount' : np.zeros((time_size, batch_size)),
      'action' : np.zeros((time_size, batch_size), dtype=np.int32),
    }
    data = collections.namedtuple('data', data.keys())(**data)

    aux_loss = usfa.QLearningAuxLoss(coeff=1.0, discount=.99)

    data, preds = jax.tree_map(lambda a:jnp.array(a), (data, preds))

    loss, metrics = aux_loss(data, preds, preds)
    print(metrics)

if __name__ == '__main__':
  absltest.main()
