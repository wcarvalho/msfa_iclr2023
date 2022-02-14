"""Tests for utils."""

from absl.testing import absltest

import haiku as hk
import chex
import jax
import jax.numpy as jnp
import numpy as np

chex.set_n_cpu_devices(4)

from modules.embedding import OneHotTask, OAREmbedding

class EmbeddingTest(absltest.TestCase):

  def test_onehot_single(self):
    raw_size=4
    latent_size=32

    w_init = np.random.rand(raw_size, latent_size)

    def network(x):
      embedder = OneHotTask(raw_size, latent_size, embedding_matrix=w_init)
      return jax.vmap(embedder)(x)

    khot = np.zeros(shape=(raw_size, raw_size))
    khot[np.arange(raw_size), np.arange(raw_size)] = 1

    net = hk.without_apply_rng(hk.transform(network))

    params = net.init(jax.random.PRNGKey(42), khot)
    fn = jax.jit(net.apply)
    y = fn(params, khot)

    np.testing.assert_allclose(w_init, y)

  def test_onehot_multi(self):
    raw_size=4
    latent_size=32

    w_init = np.random.rand(raw_size, latent_size)

    def network(x):
      embedder = OneHotTask(raw_size, latent_size, embedding_matrix=w_init)
      return jax.vmap(embedder)(x)

    khot = np.zeros(shape=(1, raw_size))
    khot[:, [1,2]] = 1

    net = hk.without_apply_rng(hk.transform(network))

    params = net.init(jax.random.PRNGKey(42), khot)
    fn = jax.jit(net.apply)
    y = fn(params, khot)

    np.testing.assert_allclose(w_init[1] + w_init[2], y[0])

if __name__ == '__main__':
  absltest.main()
