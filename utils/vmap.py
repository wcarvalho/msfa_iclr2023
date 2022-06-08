import jax
import jax.numpy as jnp
import haiku as hk
import functools
from pprint import pprint

def multihead_switch(x, fn):
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

def multihead_lift(x, fn):
  """create N parallel copies of fn with their own parameters.

  Args:
      fn (TYPE): function to copy
      x (TYPE): data (N x D). N = heads. D = dim of data.

  Returns:
      TYPE: Description
  
  Raises:
      RuntimeError: error if x is not jnp.ndarray
  """
  # assert len(x.shape)==2, "only know how to deal with N x D"

  # inner function will vmap over dimension N
  N = x.shape[0]
  functions = [fn() for i in jnp.arange(N)]

  # nested transforms for all 8 + extract inits + applies
  init_applies = jax.tree_map(lambda net: hk.transform(net), functions)  # nested transform
  inits = [i.init for i in init_applies]

  # get lifted parameters for each linear
  params = [hk.lift(inits[i], name=f"lift")(hk.next_rng_key() if hk.running_init() else None, x) for i in range(N)]

  def rename(pname, fn, fn0):
    """ rename pname so uses same base as fn0
    """
    name_pieces = pname.split("/")
    fn0_pieces = fn0.module_name.split("/")
    idx = len(fn0_pieces)
    name_pieces[:idx] = fn0_pieces
    return "/".join(name_pieces)

  # print('-----original-----')
  # pprint(jax.tree_map(lambda x:x.shape, params))

  unified_params = [{rename(k, fn, functions[0]) :v for k,v in p.items()} for p, fn in zip(params, functions)]
  # stack params
  stacked_params = jax.tree_map(lambda *arrays: jnp.stack(arrays), *unified_params)


  def apply_fn(params, x):
      key = hk.next_rng_key()
      # since using apply from 1st function, use names (see rename) from 1st function
      return init_applies[0].apply(params, key, x)

  # print('-----unified_params-----')
  # pprint(jax.tree_map(lambda x:x.shape, unified_params))
  # print('-----stacked_params-----')
  # pprint(jax.tree_map(lambda x:x.shape, stacked_params))

  x = jax.vmap(apply_fn, in_axes=(0, 0), out_axes=0)(stacked_params, x)

  return x

def batch_multihead(x : jnp.ndarray, fn : hk.Module, wrap_vmap=lambda x:x, vmap: str='lift'):
  """See multihead_{lift/switch}. Wrapper for B x N x D data.
  
  Args:
      x (jnp.ndarray): data
      fn (hk.Module): function to make apply in parallel
      vmap (str, optional): 
  
  Returns:
      TYPE: Description
  """
  # vmap over dimension 0 = batch
  # only know how to vmap over data. make fn part of function specification
  assert vmap in ['lift', 'switch', 'off'], "vmap must be {lift, off, switch}"

  if vmap == 'lift':
    # more pure jax
    vmap_func = functools.partial(multihead_lift, fn=fn)
    vfn = hk.vmap(vmap_func, in_axes=0, split_rng=False)
    vfn = wrap_vmap(vfn)
    return vfn(x)
  elif vmap == 'switch':
    # leverages haiku functions more
    vmap_func = functools.partial(multihead_switch, fn=fn)
    vfn = hk.vmap(vmap_func, in_axes=0)
    vfn = wrap_vmap(vfn)
    return vfn(x)
  else:
    N = x.shape[1]
    outs = []
    for idx in jnp.arange(N):
      outs.append(fn()(x[:, idx]))
    return jnp.stack(outs, axis=1)
