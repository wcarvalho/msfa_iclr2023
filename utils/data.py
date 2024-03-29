from typing import NamedTuple, Optional, Tuple, List, Sequence, Dict

import dataclasses

import json
import collections
import jax
import jax.numpy as jnp

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
# configs
# ======================================================

def save_dict(dictionary, file, **kwargs):
  with open(file, 'w') as handle:
    new = dictionary
    new.update(kwargs)
    def fits(x):
      y = isinstance(x, str)
      y = y or isinstance(x, float)
      y = y or isinstance(x, int)
      y = y or isinstance(x, bool)
      return y
    new = {k:v for k,v in new.items() if fits(v)}
    json.dump(new, handle)
    return new

def merge_configs(dataclass_configs, dict_configs):

  if not isinstance(dataclass_configs, list):
    dataclass_configs = [dataclass_configs]
  if not isinstance(dict_configs, list):
    dict_configs = [dict_configs]

  everything = {}
  for tc in dataclass_configs:
    everything.update(tc.__dict__)

  for dc in dict_configs:
    everything.update(dc)

  config = dataclass_configs[0]
  for k,v in everything.items():
    setattr(config, k, v)

  return config

# ======================================================
# handling tensors
# ======================================================
def expand_tile_dim(x, size, axis=-1):
  """E.g. shape=[1,128] --> [1,10,128] if dim=1, size=10
  """
  ndims = len(x.shape)
  _axis = axis
  if axis < 0: # go AFTER -axis dims, e.g. x=[1,128], axis=-2 --> [1,10,128]
    axis += 1
    _axis = axis % ndims # to account for negative

  x = jnp.expand_dims(x, _axis)
  tiling = [1]*_axis + [size] + [1]*(ndims-_axis)
  return jnp.tile(x, tiling)

def meshgrid(x: jnp.ndarray, y: jnp.ndarray):
  """concatenate all pairs
  Args:
      x (jnp.ndarray): N x D1
      y (jnp.ndarray): M x D2
  
  Returns:
      jnp.ndarray: (N x M) x (D1 + D2)
  """
  x2 = jnp.tile(x, [len(y), 1])
  y2 = jnp.repeat(y, len(x), axis=0)
  return jnp.concatenate((x2, y2), axis=-1)
