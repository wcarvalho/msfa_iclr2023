import jax
import jax.numpy as jnp
import haiku as hk
import functools

def multihead(x, fn):
    """create N parallel copies of fn with their own parameters.
    supports functions that return multiple outputs as tuples or list.

    Args:
        fn (TYPE): function to copy
        x (TYPE): data (N x D). N = heads. D = dim of data.

    Returns:
        TYPE: Description
    
    Raises:
        RuntimeError: error if x is not jnp.ndarray
    """
    assert len(x.shape)==2, "only know how to deal with N x D"


    # inner function will vmap over dimension N
    N = x.shape[0]
    functions = [fn() for i in jnp.arange(N)]
    if hk.running_init():
      # during initialization, just create functions
      example = x[0]
      # reuse same example for all functions
      x = [f(example) for f in functions]

      # combine all outputs at leaf jnp.ndarray level
      # return jax.tree_map(lambda *arrays: jnp.stack(arrays), *x)
      return jnp.stack(x)
    else:
      index = jnp.arange(N)
      # during apply, apply functions in parallel
      vmap_functions = hk.vmap(lambda i, x: hk.switch(i, functions, x), split_rng=True)
      x = vmap_functions(index, x)

    return x


def batch_multihead(x, fn):
    """See multihead. Wrapper for B x N x D data.
    """
    # vmap over dimension 0 = batch
    # only know how to vmap over data. make fn part of function specification
    vmap_func = functools.partial(multihead, fn=fn)
    return hk.vmap(vmap_func, in_axes=0)(x)

