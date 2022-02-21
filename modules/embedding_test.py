"""Tests for utils."""

from absl.testing import absltest

import rlax
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

  def test_onehot_gradient(self):
    raw_size=4
    latent_size=32

    w_init = np.random.rand(raw_size, latent_size)
    normalize = True
    def network(x):
      embedder = OneHotTask(raw_size, latent_size, embedding_matrix=w_init)
      w_embed = jax.vmap(embedder)(x) # [B, D]
      if normalize:
        w_embed/(1e-5+jnp.linalg.norm(w_embed, axis=-1, keepdims=True))
      return w_embed

    khot = np.zeros(shape=(raw_size, raw_size))
    khot[np.arange(raw_size), np.arange(raw_size)] = 1
    khot = np.concatenate((khot, np.zeros(shape=(raw_size, raw_size))))

    net = hk.without_apply_rng(hk.transform(network))

    params = net.init(jax.random.PRNGKey(42), khot)

    net = jax.jit(net.apply)

    def loss_fn(params, x):

      task = net(params, x)
      rewards = jnp.zeros(shape=(2*raw_size))

      cumulants = jnp.ones(shape=(2*raw_size, latent_size))
      reward_pred = jnp.sum(cumulants*task, -1)

      error = rlax.l2_loss(predictions=reward_pred, targets=rewards)

      print(error.sum(-1))
      return error.mean()

    error, grads = jax.value_and_grad(
      loss_fn)(params, khot)

    print(grads)
    # assert np.isnan(y.sf).any() == False
    # error = loss(params, khot)
    import ipdb; ipdb.set_trace()
    # np.testing.assert_allclose(w_init, y)


if __name__ == '__main__':
  absltest.main()
