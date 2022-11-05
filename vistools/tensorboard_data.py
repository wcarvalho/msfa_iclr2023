import json
import sys
from multiprocessing.pool import ThreadPool
import pandas as pd
import numpy as np
from pprint import pprint
import glob
import os
from tqdm import tqdm
import copy
import collections
import re
from IPython.display import display

from sklearn import metrics

def flatten_dict(d, parent_key='', sep='_'):
  items = []
  for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, collections.MutableMapping):
          items.extend(flatten_dict(v, new_key, sep=sep).items())
      else:
          items.append((new_key, v))
  return dict(items)

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class FalseDict(object):
    def __getitem__(self,key):
        return 0
    def __contains__(self, key):
        return True

def load_settings(experiment_path, basepath, **kwargs):
    """Summary

    Args:
            experiment_path (TYPE): Description
            basepath (TYPE): Description
            setting_split (TYPE): Description
            **kwargs: Description

    Returns:
            TYPE: Description

    """
    path = experiment_path.split(basepath)[1]
    experiment_settings=",".join(path.split("/")[:-1])
    experiment_settings_seed=",".join(path.split("/"))

    return dict(
      path=path,
      fullpath=experiment_path,
      experiment_settings=experiment_settings,
      experiment_settings_seed=experiment_settings_seed,
    )

def glob_format(path):
    return path.replace("['", "[[]'").replace("']", "'[]]")

class TensorboardData(object):
    """docstring for TensorboardData"""

    def __init__(self,
                 search_paths=None,
                 search_kwargs=None,
                 path_information=None,
                 settings_df=None,
                 log_events=None,
                 data=None,
                 data_df=None,
                 config_file='config.pickle',
                 ):

        self.collected_seeds = set()
        self.log_events = log_events or collections.defaultdict(list)
        self.data_df = data_df or {}
        self.config_file = config_file
        self.data = data or {}
        self.settings_df = settings_df if settings_df is not None else pd.DataFrame()

        self.path_information = path_information  or {}   # will be populate by load_paths

        self.search_paths = search_paths or []
        self.search_kwargs = search_kwargs or {}
        if search_paths:
            self.load_paths()


    def load_paths(self, search_paths=None, search_kwargs=None, config_file=None):

        search_paths = search_paths or []
        search_kwargs = search_kwargs or dict()
        config_file = config_file or self.config_file
        # new list of search paths to reload
        self.search_paths = list(set(self.search_paths+search_paths))

        # update reload settings
        self.search_kwargs = search_kwargs if search_kwargs else self.search_kwargs


        path_infos = load_path_info(
                search_paths=self.search_paths, **self.search_kwargs)

        # ======================================================
        # add event accumulators for each log event
        # ======================================================
        new_path_info = []


        for path_info in path_infos:
            path = path_info['fullpath']
            experiment_settings = path_info['experiment_settings']
            experiment_settings_seed = path_info['experiment_settings_seed']
            # path_info['experiment_settings'] = experiment_settings
            # settings = path_info['settings']

            # use this to keep track of path information
            self.path_information[path] = path_info

            os.path.join(path.replace("'", "\'"), "*")

            for subpath_info in os.walk(path):
              run = subpath_info[0]
              if os.path.isdir(run):
                checkpoint_file = os.path.join(run, config_file)
                checkpoint_exists = os.path.exists(checkpoint_file)
                if checkpoint_exists:
                  # keep track of new path information
                  if not experiment_settings in self.log_events:
                      new_path_info.append(path_info)

                  if not experiment_settings_seed in self.collected_seeds:
                    accumulator = EventAccumulator(run, size_guidance=FalseDict())
                    self.log_events[experiment_settings].append(accumulator)
                    self.collected_seeds.add(experiment_settings_seed)


        # ======================================================
        # add to dataframe
        # ======================================================
        print(f"Added: {len(new_path_info)}")
        df = pd.DataFrame(new_path_info)
        self.settings_df = pd.concat((self.settings_df, df))


    def load_settings(self, skip=[], config_search='config.json', show_varied=True):
        """For each experiment, load the settings (e.g. configs). 
        
        Args:
            skip (list, optional): Description
            config_search (str, optional): search pattern to use to look for configs.
        """
        if isinstance(skip, str): skip = [skip]

        # pick a random setting inside experiment path
        example = os.path.join(self.paths[0], config_search)
        assert os.path.exists(example), "path seems wrong"

        setting_files = [glob.glob(glob_format(os.path.join(p, config_search)))[0] for p in self.paths]

        # ======================================================
        # first get all the possible keys to get
        # ======================================================
        possible_keys = set()
        flat_configs = []
        for setting_file in setting_files:
            with open(setting_file, 'r') as f:
                config = json.load(f)
                flat_config = flatten_dict(config, sep=":")
                flat_configs.append(flat_config)

                to_add = []
                for k in flat_config.keys():
                    skip_k = sum([s in k for s in skip]) > 0
                    if not skip_k:
                        to_add.append(k)
                possible_keys.update(to_add)

        # ======================================================
        # fetch settings from all configss
        # ======================================================
        fetched_settings = {k:[] for k in possible_keys}
        for flat_config in flat_configs:
            for k in possible_keys:
                value = flat_config.get(k, None)
                fetched_settings[k].append(value)

        # ======================================================
        # add to pandas which stores
        # ======================================================
        for k in possible_keys:
            self.settings_df[k] = fetched_settings[k]

        if show_varied:
          self.varied(show=True)

    def load_tensorboard(self, njobs=4, load_paths=False, key_filter=None, val_fn=None):
        """Summary
        
        Args:
            njobs (int, optional): Description
            load_paths (bool, optional): Description
            key_filter (None, optional): Description
        """
        if load_paths:
            self.load_paths()
            self.load_settings(show_varied=False)
        # -----------------------
        # clear data statistics dataframe
        # clear previous data arrays being stored
        # will be updated with refreshed data
        # -----------------------
        self.data = collections.defaultdict(dict)



        for setting, log_events in tqdm(self.log_events.items(), desc='settings'):

            # -----------------------
            # load log events in parallel (faster)
            # -----------------------
            tbkey__to__arrays_list = multiprocess_reload_event_accumulators(
                    key_filter=key_filter,
                    val_fn=val_fn,
                    log_events=log_events,
                    njobs=njobs)
            for tbkey__to__arrays in tbkey__to__arrays_list:
                for tbkey, tbdata in tbkey__to__arrays.items():
                    if setting in self.data[tbkey]:
                        self.data[tbkey][setting].append(tbdata)
                    else:
                        self.data[tbkey][setting] = [tbdata]

        # ======================================================
        # now, go through all data, turn it into tensors, and record statistics
        # ======================================================
        self.value_types = list(self.data.keys())
        for value_type in self.data.keys():
            nonzeros = []
            means = []
            areas = []
            maxes = []
            mins = []
            num_seeds = []

            experiment_settings = list(self.data[value_type].keys())
            if len(experiment_settings) == 0:
                print("="*25)
                print("\tWARNING: %s empty" % value_type)
                print("="*25)
                self.data[value_type] = {setting: [] for setting in self.log_events.keys()}
                self.data_df[value_type] = None
                continue

            for setting in self.data[value_type].keys():
                full_arrays = self.data[value_type][setting]
                lengths = [len(a) for a in full_arrays]
                min_length = min(lengths)
                arrays = [a[:min_length] for a in full_arrays]
                arrays = np.array(arrays)
                self.data[value_type][setting] = full_arrays

                mean_array = arrays.mean(0)
                means.append(mean_array.mean())
                maxes.append(mean_array.max())
                mins.append(mean_array.min())
                try:
                    areas.append(metrics.auc(np.arange(len(mean_array)), mean_array))
                except ValueError as ve:
                    areas.append(0)
                num_seeds.append(len(arrays))
                nonzeros.append((arrays.sum(-1) > 0).sum())

            self.data_df[value_type] = pd.DataFrame.from_dict(dict(
                    experiment_settings=experiment_settings,
                    max=maxes,
                    mean=means,
                    area=areas,
                    min=mins,
                    num_seeds=num_seeds,
                    nonzero=nonzeros,
            ))
        print("Keys:")
        pprint(self.value_types)

    def subset(self, key, value):
        return self.settings[self.settings[key] == value]

    def varied(self, settings=None, show=True):
      settings = settings or []
      columns = self.settings_df.columns
      columns = [c for c in columns if not c in ['path', 'fullpath', 'experiment_settings', 'experiment_settings_seed']]
      vals = {c: set([u for u in self.settings_df[c].unique() if u==u and u is not None]) - set([None, np.nan]) for c in columns}
      unique = {c:v for c,v in vals.items() if len(v) > 1}
      unique_none = {c:None for c,v in vals.items() if len(v) > 1}

      settings = list(set(settings).union(set(unique.keys())))
      df = self.settings_df[settings]
      if show:
        pprint(unique)
        display(df)

      return unique_none, unique, df
    
    def filter(self, filters):
        outputs = []
        for filter_dict in filters:
            filter_df = conditions_met_in_df(self.settings_df, filter_dict)

            filtered_settings_df = self.settings_df[filter_df]

            if filtered_settings_df.empty:
                print("No Matches for", filter_dict)
                outputs.append(None)
                continue

            filtered_settings = filtered_settings_df['experiment_settings'].tolist()
            outputs.append(self.create_filtered_run(filtered_settings))

        return outputs

    def merge(self, filters, verbosity=0):
        outputs = []
        for filter_dict in filters:
            filter_df = conditions_met_in_df(self.settings_df, filter_dict)

            filtered_settings_df = self.settings_df[filter_df]

            if filtered_settings_df.empty:
                if verbosity:
                    print("No Matches for", filter_dict)
                outputs.append(None)
                continue
            else:
                if verbosity > 1:
                    print("Matches for", filter_dict)

            filtered_settings = filtered_settings_df['experiment_settings'].tolist()
            outputs.append(self.create_merged_run(filtered_settings))

        return outputs

    def filter_topk(self, key, filters, topk=1, column='max', bigger_is_better=True, verbose=False, return_settings=False):
        outputs = []
        if not key in self.data_df:
            if verbose:
                print(f"`{key}` not available. Options:")
                pprint(self.value_types)
            return outputs

        all_filtered_settings = []
        for filter_dict in filters:
          filter_df = conditions_met_in_df(self.settings_df, filter_dict)

          filtered_settings_df = self.settings_df[filter_df]
          if filtered_settings_df.empty:
              if verbose:
                  print("No Matches for", filter_dict)
              outputs.append(None)
              continue

          filtered_settings = filtered_settings_df['experiment_settings'].tolist()

          key_filtered_data_df = self.data_df[key][self.data_df[key]['experiment_settings'].isin(
                  filtered_settings)]

          if topk and topk < len(filtered_settings):
              if bigger_is_better:
                  key_filtered_data_df = key_filtered_data_df.nlargest(topk, column)
                  key_filtered_data_df.sort_values(
                          column, ascending=True, inplace=True)
              else:
                  key_filtered_data_df = key_filtered_data_df.nsmallest(topk, column)
                  key_filtered_data_df.sort_values(
                          column, ascending=False, inplace=True)

              filtered_settings = key_filtered_data_df['experiment_settings'].tolist()

          outputs.append(self.create_filtered_run(filtered_settings))
          all_filtered_settings.extend(filtered_settings)

          if return_settings:
              return outputs, all_filtered_settings
          else:
              return outputs

    def create_filtered_run(self, filtered_settings):

        filtered_data = {value_type: 
            { setting: 
                self.data[value_type][setting] for setting in self.data[value_type].keys(
                                         ) if setting in filtered_settings}
                                         for value_type in self.value_types}

        filtered_data_df = {value_type: self.data_df[value_type][self.data_df[value_type]['experiment_settings'].isin(
                filtered_settings)] if not self.data_df[value_type].empty else None for value_type in self.value_types}

        filtered_settings_df = self.settings_df[self.settings_df['experiment_settings'].isin(
                filtered_settings)]

        filtered_log_events = {
                setting: self.log_events[setting] for setting in filtered_settings}

        filtered_run = TensorboardData(
                # value_types=self.value_types,
                # shallow copies
                path_information={},
                log_events=copy.copy(filtered_log_events),
                settings_df=copy.copy(filtered_settings_df),
                data=copy.copy(filtered_data),
                data_df=copy.copy(filtered_data_df),
        )
        return filtered_run

    def create_merged_run(self, filtered_settings):

        # ======================================================
        # merge the data
        # ======================================================
        filtered_data = {}
        setting = None
        for value_type in self.value_types:
            arrays = []
            for setting in self.data[value_type].keys():
                if not setting in filtered_settings:
                    continue
                # import ipdb; ipdb.set_trace()
                arrays.extend(self.data[value_type][setting])

            # lengths = []
            # for set in arrays:
            #     for s in set:
            #             lengths.append(s.shape[0])
            # import ipdb; ipdb.set_trace()
            # lengths = [a.shape[1] for a in arrays]
            # min_length = min(lengths)
            # joint = np.concatenate([a[:,:min_length] for a in arrays])
            filtered_data[value_type] = {setting: arrays}

        filtered_data_df = {}
        for value_type in self.value_types:
            filtered_data_df[value_type] = self.data_df[value_type][self.data_df[value_type]['experiment_settings'].isin([setting])]

        # filtered_data_df = {value_type: self.data_df[value_type][self.data_df[value_type]['experiment_settings'].isin(filtered_settings)] if not self.data_df[value_type].empty else None for value_type in self.value_types}

        filtered_settings_df = self.settings_df[self.settings_df['experiment_settings'].isin([setting])]

        filtered_log_events = {setting: self.log_events[setting]}

        filtered_run = TensorboardData(
                value_types=self.value_types,
                # shallow copies
                path_information={},
                log_events=copy.copy(filtered_log_events),
                settings_df=copy.copy(filtered_settings_df),
                data=copy.copy(filtered_data),
                data_df=copy.copy(filtered_data_df),
        )
        return filtered_run

    def __getitem__(self, key):
        return self.data_df[key], self.data[key]

    def keys_like(self, *names):
        matches = []
        keys = {k.lower(): k for k in self.data.keys()}
        for name in names:
            found = list(filter(lambda k: len(re.findall(name, k)) > 0,  keys.keys()))
            matches.extend([keys[f] for f in found])
        return matches

    @property
    def paths(self):
        return self.settings_df['fullpath'].tolist()

    @property
    def keys(self):
        return [k.lower() for k in self.data.keys()]


# ======================================================
# Utility functions
# ======================================================

# -----------------------
# for getting filters over dataframes
# -----------------------

# -----------------------
# for getting matches in dataframe
# -----------------------
def condition_met_in_df(df, key, value):
    if value is None or value!=value or (isinstance(value, str) and value.lower() == 'none'):
        x = df[key] == 'none'
        y = df[key].isnull()
        return x | y
    elif isinstance(value, list) or isinstance(value, np.ndarray):
        vals_in_df = df[key].tolist()
        matches = [v == value for v in vals_in_df]
        return np.array(matches)
    else:
        return df[key] == value


def conditions_met_in_df(df, conditions):
    conditions = iter(conditions.items())
    key, value = next(conditions)

    met = condition_met_in_df(
            df, key=key, value=value)

    try:
        for k, v in conditions:
            current_met = condition_met_in_df(df, key=k, value=v)
            met = met & current_met
    except StopIteration as se:
        pass
    except Exception as e:
        raise e

    return met


# ======================================================
# for loading tensorboard data
# ======================================================
# -----------------------
# loading data
# -----------------------
def multiprocess_reload_event_accumulators(log_events, njobs=16, 
    key_filter=None,
    val_fn=None):
    with ThreadPool(processes=njobs) as pool:
        futures = []
        for log_event in log_events:
            future = pool.apply_async(reload_event_accumulator, [log_event, key_filter, val_fn])
            futures.append(future)

        outputs = []
        for future in futures:
            output = future.get()
            if output:
                outputs.append(output)

        return outputs


def reload_event_accumulator(log_event, key_filter=None, val_fn=None):
    log_event.Reload()
    val_fn = val_fn or tf.make_ndarray
    # ======================================================
    # load all data from log path to data objects
    # ======================================================

    key2val = {}
    # key2step = {}

    keys = log_event.Tags()['tensors']

    if key_filter:
        keys = filter(key_filter, keys)

    for key in keys:
        try:
            w_times, step_nums, vals = zip(*log_event.Tensors(key))
        except:
            key2val[key] = np.array([-np.inf])
            continue

        # ======================================================
        # store {setting: [values]}
        # ======================================================
        key2val[key] = np.array([val_fn(v) for v in vals])
        key2val[f"{key}_steps"] = np.array(step_nums)

    return key2val



def load_path_info(
    basepath,
    search_paths,
    verbose=1,
    njobs=8,
    **kwargs
    ):

    full_search_paths = [os.path.join(basepath, p) for p in search_paths]

    def handle_p(p):
        settings = load_settings(
                experiment_path=p,
                basepath=basepath)
        return p, settings

    outputs = []
    loaded = set()
    with ThreadPool(processes=njobs) as pool:
        futures = []
        for search_path in full_search_paths:
            if verbose:
                sys.stderr.write(f"get_runs: {str(search_path)}\n")
                sys.stderr.flush()

            experiment_paths = list(sorted(glob.glob(search_path)))

            experiment_paths = [p for p in experiment_paths if os.path.isdir(p)]
            max_len = len(experiment_paths)


            if verbose:
                print("="*25)
                print("%s: %d" % (search_path, max_len))
                print("="*25)
            for p in experiment_paths[:max_len]:

                future = pool.apply_async(handle_p, [p])
                futures.append(future)

        hits = 0
        for future in tqdm(futures, desc="loaded"):
            p, output = future.get()
            if p:
                hits += 1
                loaded.add(p)
                outputs.append(output)
                if verbose > 1:
                    print("loaded", p)

    if not hits:
        print("="*25)
        print(search_paths)
        print("NO PATHS FOUND")
        print("="*25)

    return outputs