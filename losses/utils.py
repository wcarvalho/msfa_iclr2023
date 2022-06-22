import jax.numpy as jnp



def make_episode_mask(data, include_final=False, dtype=jnp.float32):
  """Look at where have valid task data. Everything until 1 before final valid data counts towards task. Data.discount always ends two before final data. 
  e.g. if valid data is [x1, x2, x3, 0, 0], data.discount is [1,0,0,0,0]. So can use that to obtain masks.
  
  Args:
      data (TYPE): Description
      include_final (bool, optional): if True, include all data. if False, include until 1 time-step before final data
  
  Returns:
      TYPE: Description
  """
  T, B = data.discount.shape
  if include_final:
    # return [1,1,1,0,0]
    return jnp.concatenate((jnp.ones((2, B)), data.discount[:-2]), axis=0)
  else:
    # return [1,1,0,0,0]
    return jnp.concatenate((jnp.ones((1, B)), data.discount[:-1]), axis=0)

def episode_mean(x, mask):
  if len(mask.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(mask.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    batch_loss = jnp.multiply(x, jnp.expand_dims(mask, dims))
  else:
    batch_loss = jnp.multiply(x, mask)
  return (batch_loss.sum(0))/(mask.sum(0)+1e-5)
