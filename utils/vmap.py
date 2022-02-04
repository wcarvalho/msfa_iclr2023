import jax
import jax.numpy as jnp
import haiku as hk

def multihead(fn, x, N):
    """create N parallel copies of fn with their own parameters.
    supports functions that return multiple outputs as tuples or list.
    
    
    Args:
        fn (TYPE): function to copy
        x (TYPE): data
        N (TYPE): number of heads
    
    Returns:
        TYPE: Description
    
    Raises:
        RuntimeError: error if x is not jnp.ndarray
    """
    if not isinstance(x, jnp.ndarray):
        raise RuntimeError("Don't know how to handle anything except single array input. Can do multiple outputs.")
    assert x.shape[0]==N, "make sure `x` & `N` fn are correct"

    functions = [fn() for i in range(N)]
    index = jnp.arange(len(functions))

    if hk.running_init():
        # during initialization, just create functions
        example = x[0]
        # reuse same example for all functions
        x = [f(example) for f in functions]

        # combine all outputs at leaf jnp.ndarray level
        return jax.tree_map(lambda *arrays: jnp.stack(arrays), *x)
    else:
      # during apply, apply functions in parallel
      vmap_functions = hk.vmap(lambda i, x: hk.switch(i, functions, x))
      x = vmap_functions(index, x)

    return x


def multihead_tr(x, transpose_fn, **kwargs):
    """
    transpose_fn specifies how to get dimension of heads to 0th dimension. after multihead function is made, dimensions are swapped back.
    """
    x = jax.tree_map(transpose_fn, x)
    x = multihead(x=x, **kwargs)

    x = jax.tree_map(transpose_fn, x)
    return x

