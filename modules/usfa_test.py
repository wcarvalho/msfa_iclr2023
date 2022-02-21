"""Tests for utils."""

from absl.testing import absltest

import haiku as hk
import chex
import jax
import jax.numpy as jnp
import numpy as np

chex.set_n_cpu_devices(4)

from modules.usfa import UsfaHead, UniqueStatePolicyPairs, ConcatFlatStatePolicy, CumulantsAuxTask, USFAInputs

class USFATest(absltest.TestCase):
  num_actions: int = 4
  state_dim: int = 8
  hidden_size : int=128
  policy_size : int=32
  variance: float=0.1
  nsamples: int=10
  sf_input_fn = None
  task_embed: int = 0
  normalize_task: bool = True

  def test_embed_task(self):
    task_embed_size = 128
    def network(w, memory_out):
      usfa_head = UsfaHead(
        num_actions=self.num_actions,
        state_dim=self.state_dim,
        hidden_size=self.hidden_size,
        policy_size=self.policy_size,
        variance=self.variance,
        nsamples=self.nsamples,
        sf_input_fn=self.sf_input_fn,
        task_embed=task_embed_size,
        normalize_task=self.normalize_task)
      return usfa_head(USFAInputs(w=w, memory_out=memory_out), hk.next_rng_key())

    w = np.zeros(shape=(10, self.state_dim))
    w[:, 1] = 1
    memory_out = np.random.rand(10, 512)

    net = hk.transform(network)

    params = net.init(jax.random.PRNGKey(42), w, memory_out)
    fn = jax.jit(net.apply)
    y = fn(params, jax.random.PRNGKey(42), w, memory_out)

    assert np.isnan(y.sf).any() == False
    assert np.isnan(y.q).any() == False
    assert np.isnan(y.z).any() == False
    assert np.isnan(y.w_embed).any() == False

    cumulant_dim = (10, 1+self.nsamples, self.num_actions, task_embed_size)
    self.assertEqual(y.q.shape, (10, self.num_actions))
    self.assertEqual(y.sf.shape, cumulant_dim)
    self.assertEqual(y.z.shape, cumulant_dim)
    self.assertEqual(y.w_embed.shape, (10, task_embed_size))

  def test_no_embed_task(self):
    task_embed_size = 0
    def network(w, memory_out):
      usfa_head = UsfaHead(
        num_actions=self.num_actions,
        state_dim=self.state_dim,
        hidden_size=self.hidden_size,
        policy_size=self.policy_size,
        variance=self.variance,
        nsamples=self.nsamples,
        sf_input_fn=self.sf_input_fn,
        task_embed=task_embed_size,
        normalize_task=self.normalize_task)
      return usfa_head(USFAInputs(w=w, memory_out=memory_out), hk.next_rng_key())

    w = np.random.rand(10, self.state_dim)
    memory_out = np.random.rand(10, 512)

    net = hk.transform(network)

    params = net.init(jax.random.PRNGKey(42), w, memory_out)
    fn = jax.jit(net.apply)
    y = fn(params, jax.random.PRNGKey(42), w, memory_out)

    cumulant_dim = (10, self.nsamples+1, self.num_actions, self.state_dim)

    self.assertEqual(y.q.shape, (10, self.num_actions))
    self.assertEqual(y.sf.shape, cumulant_dim)
    self.assertEqual(y.z.shape, cumulant_dim)
    self.assertEqual(y.w_embed.shape, w.shape)

    # base z is same as task
    np.testing.assert_array_almost_equal(y.z[:,0,0], w)


    assert np.isnan(y.sf).any() == False
    assert np.isnan(y.q).any() == False
    assert np.isnan(y.z).any() == False
    assert np.isnan(y.w_embed).any() == False

if __name__ == '__main__':
  absltest.main()
