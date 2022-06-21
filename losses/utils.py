import jax.numpy as jnp



def make_episode_mask(data, include_final=False, dtype=jnp.float32):
  """Look at where have valid task data. Everything until 1 before final valid data counts towards task
  
  Args:
      data (TYPE): Description
      include_final (bool, optional): if False, mask out losses based on 
  
  Returns:
      TYPE: Description
  """
  task = data.observation.observation.task
  task_mask = (task.sum((2)) > 0).astype(dtype)
  if include_final:
    return task_mask

  T, B = task.shape[:2]
  zeros = jnp.zeros((1, B), dtype=dtype)
  episode_mask = jnp.concatenate((task_mask[1:], zeros), axis=0)
  return episode_mask


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
