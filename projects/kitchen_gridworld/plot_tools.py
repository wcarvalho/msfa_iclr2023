import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

ACTIONS = ['left', 'right', 'forward', 'pickup_container', 'pickup_contents', 'place', 'toggle', 'slice']

def split_episodes(episodes):
    rewarding = []
    failure = []
    for episode in episodes:
        if episode['rewards'].sum() > 0:
          rewarding.append(episode)
        else:
          failure.append(episode)
    return dict(
      success=rewarding,
      failure=failure,
      )

# ======================================================
# General plotting
# ======================================================
def plot_image(ax, episode, tdx):
  ims = episode['observation']['image']
  ax.imshow(ims[tdx])
  ax.grid(False)
  ax.xaxis.set_ticklabels([])
  ax.yaxis.set_ticklabels([])

  ax.axis('off')

def bar_plot(ax, names=[], values=[], bar_kwargs={}, colors=None, important_names=[], important_color=None, ylabel="", title="", ylim=[], nticks=0, title_size=16, tick_text_size=16, rotation=45, remove_yticks=False, remove_xticks=False, **kwargs):
  y_pos = np.arange(len(values))

  if values is not None:
    bar = ax.bar(y_pos, values, align='center', **bar_kwargs)
  if colors:
    for child, color in zip(ax.get_children(), colors):
      child.set_color(color)

  if important_names:
    for name, child in zip(names, ax.get_children()):
      if name in important_names:
        child.set_color(important_color)

  if names:
    ax.set_xticks(y_pos)
    ax.set_xticklabels(names, rotation=rotation, fontsize=tick_text_size)
  if title:
    ax.set_title(title, fontsize=title_size)
  if ylabel:
    ax.set_ylabel(ylabel)
  if ylim:
      ax.set_ylim(*ylim)

  if nticks and ylim:
    length = ylim[1]-ylim[0]
    step = length/nticks
    ax.set_yticks(np.arange(ylim[0], ylim[1]+step, step))

  if remove_yticks:
    ax.yaxis.set_ticklabels([])
    ax.set_yticks([])
  if remove_xticks:
    ax.xaxis.set_ticklabels([])
    ax.set_xticks([])

  if values is not None:
    return bar

# ======================================================
# Agreement
# ======================================================

def get_sfz_stats(episode, config, verbosity=0):
    sf = episode['preds'].sf # [T, A, D]
    w = episode['preds'].w # [T, D]
    actions = episode['action']

    multiply = jax.vmap(jnp.multiply, in_axes=(1, None), out_axes=1)
    multiply = jax.vmap(multiply, in_axes=(1, None), out_axes=1)

    sf_ = multiply(sf, w) # [T, Z, A, D]
    q = sf_.sum(-1) # [T, Z, A]
    if verbosity:
        print('multiply:', sf_.shape, '=', sf.shape, w.shape)
    sf_ = jnp.stack(jnp.split(sf_, config['nmodules'], axis=-1), axis=3) # [T, Z, A, N, D/N]
    if verbosity:
        print('split by module:', sf_.shape)
    sfz = sf_.sum(-1).transpose(0, 1, 3, 2) # [T, Z, A, N] --> [T, Z, N, A]
    if verbosity:
        print('sfz norms:', sfz.shape)
    
    qsf = jnp.concatenate((jnp.expand_dims(q, axis=2), sfz), axis=2)
    
    return dict(sfz=sfz, q=q, qsf=qsf)

def get_sfw_stats(episode, config, verbosity=0):
    sf = episode['preds'].sf[:, 0] # [T, A, D]
    w = episode['preds'].w # [T, D]
    actions = episode['action']

    multiply = jax.vmap(jnp.multiply, in_axes=(1, None), out_axes=1)
    
    sf_ = multiply(sf, w) # [T, A, D]
    q = sf_.sum(-1) # [T, A]
    if verbosity:
        print('multiply:', sf_.shape, '=', sf.shape, w.shape)
    sf_ = jnp.stack(jnp.split(sf_, config['nmodules'], axis=-1), axis=2) # [T, A, N, D/N]
    if verbosity:
        print('split by module:', sf_.shape)
    sfw = sf_.sum(-1).transpose(0,2,1) # [T, A, N] --> [T, N, A]
    if verbosity:
        print('sfw norms:', sfw.shape)
    
    qsf = jnp.concatenate((jnp.expand_dims(q, axis=1), sfw), axis=1)
    
    return dict(sfw=sfw, q=q, qsf=qsf)

def interaction_labels(episode):
    info = episode['interaction_info']
    labels = []
    for i in info:
        if i:
            label = f"{i['action']}\n{i['object']}"
        else:
            label = ''
        labels.append(label)
    return labels

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def plot_agreement(ax, episode, ax_time=None, tdx=None, config=None, stats=None, option='w', ticksize=12, title='', title_size=16):
  actions = episode['action']

  assert option in ['w', 'z']
  if option == 'w':
    if stats is None:
      stats = get_sfw_stats(episode, config)

    module_actions = stats['sfw'].argmax(-1)

  elif option == 'z':
    if stats is None:
      stats = get_sfz_stats(episode, config)

    sfz = stats['sfz']
    module_actions = sfz.max(1).argmax(-1)

  agree = lambda a,b: a == b 
  agree = jax.vmap(agree, in_axes=(None, 1), out_axes=1)
  agreement = agree(actions, module_actions)


  # ======================================================
  # All Agreement
  # ======================================================

  ax.imshow(agreement.T, cmap = 'OrRd' , interpolation = 'nearest')

  # -----------------------
  # xlabels
  # -----------------------
  
  xlabels = interaction_labels(episode)
  ax.set_xticks(np.arange(len(xlabels)), labels=xlabels, rotation=90, fontsize=ticksize)

  # -----------------------
  # ylabels
  # -----------------------
  percent = agreement.mean(0)
  ylabels = ['Module {}: {:.2f}'.format(idx, p) for idx, p in enumerate(percent)]
  ax.set_yticks(np.arange(len(ylabels)), labels=ylabels, fontsize=ticksize)

  if title:
    title = title.replace("_", " ").title()
    ax.set_title(title, fontsize=title_size)

  if tdx is not None:
    for idx in range(agreement.shape[1]):
      if agreement[tdx, idx]:
        highlight_cell(tdx, idx, ax=ax, color='limegreen', linewidth=3)
  # ======================================================
  # timestep agreement
  # ======================================================
  if ax_time:
    ax_time.imshow(jnp.expand_dims(agreement[tdx], axis=1), cmap = 'OrRd' , interpolation = 'nearest')

    chosen_action = ACTIONS[actions[tdx]]
    ylabels = ['{}'.format(ACTIONS[ma]) for idx, ma in enumerate(module_actions[tdx])]
    ax_time.set_yticks(np.arange(len(ylabels)), labels=ylabels, fontsize=ticksize)

# ======================================================
# Q-values for chosen actions
# ======================================================

def get_all_qvalues(sf, w):
    multiply = jax.vmap(jnp.multiply, in_axes=(1, None), out_axes=1)
    q_values = jax.vmap(multiply)(sf, w)
    return q_values.sum(-1)

def get_chosen_a_qvalues(q_values, actions):
    def index(q : jnp.ndarray, a : int):
        return q[a]
    index = jax.vmap(index, in_axes=(0, None), out_axes=0)
    index = jax.vmap(index)
    return index(q_values, actions)

def get_chosen_a_sfs(sf_, z_idx_, action_):
    # sf [T, Z, A, D]
    # z_idx [T] 
    def index(sf : jnp.ndarray, i : int):
        return sf[i]
    index = jax.vmap(index)
    sfz = index(sf_, z_idx_)
    sfza = index(sfz, action_)
    return sfza

def get_module_sf_norms(sf_, w_, mods=4, norm='task'):
    if norm == 'task':
        q_ = sf_*w_
        # dims_per_mod = sf_.shape[-1]/mods
        q_ = jnp.stack(jnp.split(q_, mods, axis=-1), axis=1)
        return q_.sum(-1)
    elif norm == "l2":
        sf_ = jnp.stack(jnp.split(sf_, mods, axis=-1), axis=1)
        return jnp.linalg.norm(sf_, ord=2, axis=-1)
    else:
        raise RuntimeError(norm)

def get_chosen_a_sf_stats(episode, config, norm='task'):
    sf = episode['preds'].sf # [T, Z, A, D]
    w = episode['preds'].w # [T, D]
    actions = episode['action'] # [T]
    # actions = jnp.ones(actions.shape, dtype=actions.dtype)*5
    stats = dict()

    # all q-values [T, Z, A]
    all_q = get_all_qvalues(sf, w)

    # get q-values of chosen actions, [T, Z]
    q_chosen = get_chosen_a_qvalues(all_q, actions)
    
    # get chosen policy embedding, [T]
    z_idx = jnp.argmax(q_chosen, axis=1)

    # get sfs for chosen actions/policies [T, D]
    chosen_sfs = get_chosen_a_sfs(sf, z_idx, actions)
    # print('chosen_sfs', chosen_sfs.shape)

    # get weighted average [T, N]
    sf_norms = get_module_sf_norms(chosen_sfs, w, mods=config['nmodules'], norm=norm)
    
    return dict(
        chosen_sfs=chosen_sfs,
        sf_norms=sf_norms,
        q_chosen=q_chosen,
        all_q=all_q,
        z_idx=z_idx,
        actions=actions,
        sf=sf,
    )

def plotq_action(ax, episode, tdx, config, **kwargs):

  stats = get_chosen_a_sf_stats(episode, config)
  sf_norms = stats['sf_norms'][tdx] # [T, N]
  bar_plot(
    ax=ax,
    names=['{}'.format(idx) for idx in range(len(sf_norms))],
    values=sf_norms,
    rotation=90,
    **kwargs)