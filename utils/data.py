import collections
import jax

def flatten_dict(d, parent_key='', sep='_'):
  items = []
  for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, collections.MutableMapping):
          items.extend(flatten_dict(v, new_key, sep=sep).items())
      else:
          items.append((new_key, v))
  return dict(items)

def consolidate_dict_list(dict_list):
  """
  Convert:
      [dict(x=1, y=1), ..., dict(x=n, y=n)]
  to:
      dict(
        x=[1, ..., n],
        y=[1, ..., n],
      )

  
  Args:
      dict_list (list): list of dicts
  """
  consolidation = flatten_dict(dict_list[0], sep="/")
  consolidation = {k: [v] if not isinstance(v, list) else v for k,v in consolidation.items()}
  for next_dict in dict_list[1:]:
      fnext_dict = flatten_dict(next_dict, sep="/")
      for k, v in consolidation.items():
          newv = fnext_dict[k]
          if isinstance(newv, list):
              consolidation[k].extend(newv)
          else:
              consolidation[k].append(newv)

  return consolidation

def dictop(dictionary: dict, op, skip=[], verbose=False):
  """Apply function recursively to dictionary
  
  Args:
      dictionary (dict): dict
      op (TYPE): function to apply
      skip (list, optional): keys to skip
  
  Returns:
      TYPE: Description
  """
  if not isinstance(dictionary, dict):
    try:
        return op(dictionary)
    except Exception as e:
        if verbose:
            print(e)
        return None
  return {k: dictop(v, op, verbose=verbose) if (not (k is None or k in skip)) else v for k,v in dictionary.items()}



# ======================================================
# handling tensors
# ======================================================
def expand_tile_dim(x, dim, size):
  """E.g. shape=[1,128] --> [1,10,128] if dim=1, size=10
  """
  ndims = len(x.shape)
  x = jnp.expand_dims(x, dim)
  tiling = [1]*dim + [size] + [1]*(ndims-dim)
  return jnp.tile(x, tiling)
